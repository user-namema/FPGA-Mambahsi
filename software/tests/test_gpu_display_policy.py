"""GPU process-policy regression checks using synthetic NVML data; no GPU required."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpu_power_monitor import NVMLDevice
from benchmark_up_gpu_power import check_processes,process_boundary_summary,parse_args


class NVMLError(Exception):
    pass


def device(compute=(),graphics=(),names=None,fail_compute=False):
    def query_compute(handle):
        if fail_compute:raise NVMLError('not supported')
        return [SimpleNamespace(pid=pid) for pid in compute]
    def name(pid):
        value=(names or {}).get(pid)
        if value is None:raise NVMLError('name unavailable')
        return value
    nv=NVMLDevice.__new__(NVMLDevice)
    nv.handle='selected-gpu'
    nv.nv=SimpleNamespace(NVMLError=NVMLError,
        nvmlDeviceGetComputeRunningProcesses=query_compute,
        nvmlDeviceGetGraphicsRunningProcesses=lambda h:[SimpleNamespace(pid=pid) for pid in graphics],
        nvmlSystemGetProcessName=name)
    return nv


class DisplayPolicyTests(unittest.TestCase):
    def test_user_two_xorg_graphics_only_allowed_explicitly(self):
        nv=device(graphics=[4112,3467119],names={4112:b'/usr/lib/xorg/Xorg',3467119:'Xorg'})
        with self.assertRaises(RuntimeError):check_processes(nv,False)
        state=check_processes(nv,False,True)
        self.assertEqual(state['other_pids'],[4112,3467119])
        self.assertEqual(state['allowed_display_pids'],[4112,3467119])
        self.assertEqual(state['blocking_pids'],[])
        summary=process_boundary_summary(state,state)
        self.assertFalse(summary['exclusive_at_boundaries'])
        self.assertTrue(summary['no_other_compute_at_boundaries'])
        self.assertEqual(summary['process_condition'],'display_background_only')

    def test_compute_process_blocked_even_with_display_permission(self):
        for name in ['python','/usr/lib/xorg/Xorg']:
            nv=device(compute=[18956],graphics=[18956],names={18956:name})
            with self.assertRaises(RuntimeError):check_processes(nv,False,True)
            state=check_processes(nv,True,True)
            self.assertEqual(state['allowed_display_pids'],[])
            self.assertFalse(process_boundary_summary(state,state)['no_other_compute_at_boundaries'])

    def test_unknown_graphics_and_failed_queries_are_not_waived(self):
        for name in ['python','chrome',None]:
            nv=device(graphics=[11],names={11:name})
            with self.assertRaises(RuntimeError):check_processes(nv,False,True)
        nv=device(graphics=[11],names={11:'Xorg'},fail_compute=True)
        with self.assertRaises(RuntimeError):check_processes(nv,False,True)
        state=check_processes(nv,True,True)
        summary=process_boundary_summary(state,state)
        self.assertFalse(summary['no_other_compute_at_boundaries'])
        self.assertEqual(summary['process_condition'],'shared_or_unverified')

    def test_own_context_excluded_and_empty_gpu_remains_exclusive(self):
        nv=device(compute=[os.getpid()],graphics=[os.getpid()])
        state=check_processes(nv,False)
        self.assertEqual(state['other_pids'],[])
        self.assertTrue(process_boundary_summary(state,state)['exclusive_at_boundaries'])

    def test_after_boundary_other_compute_is_detected(self):
        before=check_processes(device(),False,True)
        after=check_processes(device(compute=[42],names={42:'python'}),True,True)
        result=process_boundary_summary(before,after)
        self.assertFalse(result['exclusive_at_boundaries'])
        self.assertFalse(result['no_other_compute_at_boundaries'])

    def test_cli_default_and_explicit_opt_in(self):
        with patch('sys.argv',['bench','--fp32-dir','/tmp/model']):
            args=parse_args();self.assertFalse(args.allow_display_processes)
        with patch('sys.argv',['bench','--fp32-dir','/tmp/model','--allow-display-processes']):
            args=parse_args();self.assertTrue(args.allow_display_processes)
            self.assertFalse(args.allow_other_processes)


if __name__=='__main__':unittest.main()
