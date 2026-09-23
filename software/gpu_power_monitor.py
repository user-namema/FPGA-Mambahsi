"""NVML telemetry and numerical integration; no torch dependency. Python >=3.8."""
import ctypes
import math
import os
from pathlib import Path
import statistics
import threading
import time


def as_text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def cuda_pci_bus_id(device_index):
    """Use the already-loaded CUDA runtime to respect CUDA_VISIBLE_DEVICES remapping."""
    maps=Path('/proc/self/maps')
    candidates=[]
    if maps.exists():
        for line in maps.read_text().splitlines():
            if '/libcudart' in line:
                path=line.split()[-1]
                if path.startswith('/') and path not in candidates:candidates.append(path)
    for path in candidates:
        lib=ctypes.CDLL(path)
        fn=lib.cudaDeviceGetPCIBusId
        fn.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int];fn.restype=ctypes.c_int
        buf=ctypes.create_string_buffer(64)
        if fn(buf,len(buf),device_index)==0:return buf.value.decode()
    raise RuntimeError('Cannot map CUDA device to PCI bus using loaded libcudart. '
                       'No NVML index fallback is used, to avoid measuring the wrong GPU.')


def integrate_power(samples, start, end):
    """Trapezoidal W*s, interpolated/clamped at measurement boundaries."""
    if end<=start or not samples:raise ValueError('Empty power window')
    pts=sorted((float(r['monotonic_s']),float(r['power_w'])) for r in samples)
    if any(not math.isfinite(t) or not math.isfinite(w) or w<0 for t,w in pts):
        raise ValueError('Invalid telemetry')
    def power(t):
        if t<=pts[0][0]:return pts[0][1]
        for (ta,wa),(tb,wb) in zip(pts,pts[1:]):
            if t<=tb:return wb if tb==ta else wa+(wb-wa)*(t-ta)/(tb-ta)
        return pts[-1][1]
    clipped=[(start,power(start))]+[(t,w) for t,w in pts if start<t<end]+[(end,power(end))]
    return sum((tb-ta)*(wa+wb)/2 for (ta,wa),(tb,wb) in zip(clipped,clipped[1:]))


class NVMLDevice:
    def __init__(self,cuda_index):
        try:import pynvml as nv
        except ImportError as e:raise RuntimeError('Install NVML Python bindings: python -m pip install nvidia-ml-py') from e
        self.nv=nv;nv.nvmlInit()
        self.pci_bus_id=cuda_pci_bus_id(cuda_index)
        self.handle=nv.nvmlDeviceGetHandleByPciBusId(self.pci_bus_id.encode())
        self.metadata=dict(cuda_index=cuda_index,pci_bus_id=self.pci_bus_id,
            uuid=as_text(nv.nvmlDeviceGetUUID(self.handle)),
            name=as_text(nv.nvmlDeviceGetName(self.handle)),
            driver=as_text(nv.nvmlSystemGetDriverVersion()),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            power_scope='NVML device including associated circuitry; not per-process or wall-plug')
        try:self.metadata['power_limit_w']=nv.nvmlDeviceGetPowerManagementLimit(self.handle)/1000.
        except nv.NVMLError:self.metadata['power_limit_w']=None
        self.energy_unavailable_reason=None
        self.read() # fail early if power telemetry is unsupported

    def read(self):
        n,h=self.nv,self.handle
        before=time.perf_counter();power=n.nvmlDeviceGetPowerUsage(h)/1000.;after=time.perf_counter()
        row=dict(monotonic_s=(before+after)/2,wall_time_s=time.time(),power_w=power)
        for name,fn in [('temperature_c',lambda:n.nvmlDeviceGetTemperature(h,n.NVML_TEMPERATURE_GPU)),
                        ('sm_clock_mhz',lambda:n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_SM)),
                        ('memory_clock_mhz',lambda:n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_MEM)),
                        ('gpu_util_percent',lambda:n.nvmlDeviceGetUtilizationRates(h).gpu)]:
            try:row[name]=fn()
            except n.NVMLError:row[name]=None
        return row

    def energy(self):
        try:return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)/1000.
        except self.nv.NVMLError as e:
            self.energy_unavailable_reason=str(e);return None

    def other_processes(self):
        found=set();unavailable=[];by_kind={};names={};name_errors={}
        for kind in ['Compute','Graphics']:
            pids=set()
            try:
                for p in getattr(self.nv,'nvmlDeviceGet'+kind+'RunningProcesses')(self.handle):
                    if p.pid!=os.getpid():pids.add(int(p.pid))
            except self.nv.NVMLError as e:unavailable.append(kind+': '+str(e))
            by_kind[kind]=pids;found.update(pids)
        for pid in sorted(found):
            try:names[str(pid)]=as_text(self.nv.nvmlSystemGetProcessName(pid))
            except (self.nv.NVMLError,AttributeError) as e:
                names[str(pid)]=None;name_errors[str(pid)]=str(e)
        # Only known display servers with a graphics-only context qualify.
        # A compute context always wins, even if the executable is named Xorg.
        display=[pid for pid in by_kind['Graphics']-by_kind['Compute']
                 if names[str(pid)] and Path(names[str(pid)]).name in {'Xorg','Xwayland'}]
        return dict(other_pids=sorted(found),unavailable=unavailable,
                    compute_pids=sorted(by_kind['Compute']),graphics_pids=sorted(by_kind['Graphics']),
                    display_pids=sorted(display),process_names=names,process_name_errors=name_errors)

    def close(self):self.nv.nvmlShutdown()


class PowerSampler:
    def __init__(self,device,interval=0.1):
        if interval<=0:raise ValueError('interval must be positive')
        self.device=device;self.interval=interval;self.rows=[];self.error=None
        self.event=threading.Event();self.thread=None

    def start(self):
        self.rows.append(self.device.read())
        def loop():
            try:
                while not self.event.wait(self.interval):self.rows.append(self.device.read())
            except BaseException as e:self.error=e;self.event.set()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();return self

    def stop(self):
        self.event.set()
        if self.thread:self.thread.join(timeout=5)
        if self.thread and self.thread.is_alive():raise RuntimeError('NVML sampler did not stop')
        if self.error:raise RuntimeError('Power sampling failed') from self.error
        self.rows.append(self.device.read())
        return self.rows


def energy_summary(rows,t0,t1,e0,e1,tiles):
    if tiles<=0:raise ValueError('No completed tiles')
    duration=t1-t0
    integrated=integrate_power(rows,t0,t1)
    use_counter=e0 is not None and e1 is not None and math.isfinite(e1-e0) and e1>e0
    joules=e1-e0 if use_counter else integrated
    samples=[r['power_w'] for r in rows if t0<=r['monotonic_s']<=t1]
    return dict(duration_s=duration,tiles=tiles,tiles_per_s=tiles/duration,
        energy_j=joules,energy_method='nvml_energy_counter' if use_counter else 'nvml_power_trapezoid',
        energy_counter_start_j=e0,energy_counter_end_j=e1,
        power_integrated_energy_j=integrated,mean_power_w=joules/duration,
        joules_per_tile=joules/tiles,microjoules_per_tile=joules/tiles*1e6,
        power_samples_inside=len(samples),
        sampled_min_power_w=min(samples) if samples else None,
        sampled_max_power_w=max(samples) if samples else None,
        counter_vs_integral_relative_difference=(joules-integrated)/integrated if use_counter and integrated>0 else None)
