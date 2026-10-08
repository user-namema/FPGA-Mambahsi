"""XDMA dual-64-KiB bank test; Python standard library only.

Uses the same SetupAPI -> h2c_0/c2h_0 -> seek/WriteFile/ReadFile path
as 3_pcie_memory_rw. No GPIO, DDR, inference or board programming.
--self-test does not enumerate or access hardware.
"""
import argparse
import ctypes as C
import datetime
import json
import os
from pathlib import Path
import struct
import sys
import time
import zlib

BANK_SIZE = 65536
TOTAL_SIZE = 2 * BANK_SIZE
TILE_SIZE = 4096


def check_range(address, size):
    if address < 0 or size <= 0 or address + size > TOTAL_SIZE:
        raise ValueError("DMA outside the dual-bank 0x00000000..0x0001FFFF window")
    if address % 32 or size % 32:
        raise ValueError("This test uses 32-byte-aligned AXI beats only")


def pattern(bank, generation):
    # Address-dependent words distinguish tiles, 32-KiB halves and the banks.
    data = bytearray(BANK_SIZE)
    state = (0x9E3779B9 ^ (bank * 0x1234567) ^ generation) & 0xFFFFFFFF
    for offset in range(0, BANK_SIZE, 4):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        word = (state ^ (bank * BANK_SIZE + offset) ^ (generation << 16)) & 0xFFFFFFFF
        struct.pack_into('<I', data, offset, word)
    for tile in range(16):
        struct.pack_into('<IIII', data, tile * TILE_SIZE,
                         0x42414E4B, bank, tile, generation)
    return bytes(data)


class WindowsXDMA:
    def __init__(self, device_index=None):
        if os.name != 'nt':
            raise RuntimeError('Hardware mode requires Windows and the XDMA driver')
        from ctypes import wintypes as W
        self.k = C.WinDLL('kernel32', use_last_error=True)
        self.s = C.WinDLL('setupapi', use_last_error=True)
        self.handles = []
        self.buffer = None
        self.k.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, C.c_void_p,
                                       W.DWORD, W.DWORD, W.HANDLE]
        self.k.CreateFileW.restype = W.HANDLE
        self.k.CloseHandle.argtypes = [W.HANDLE]
        self.k.SetFilePointerEx.argtypes = [W.HANDLE, C.c_longlong,
                                           C.POINTER(C.c_longlong), W.DWORD]
        self.k.SetFilePointerEx.restype = W.BOOL
        for name in ('ReadFile', 'WriteFile'):
            f = getattr(self.k, name)
            f.argtypes = [W.HANDLE, C.c_void_p, W.DWORD,
                          C.POINTER(W.DWORD), C.c_void_p]
            f.restype = W.BOOL
        self.k.VirtualAlloc.argtypes = [C.c_void_p, C.c_size_t, W.DWORD, W.DWORD]
        self.k.VirtualAlloc.restype = C.c_void_p
        self.k.VirtualFree.argtypes = [C.c_void_p, C.c_size_t, W.DWORD]
        self.paths = self.enumerate()
        if device_index is None:
            if len(self.paths) != 1:
                raise RuntimeError(f'Found {len(self.paths)} devices; use --list and --device-index')
            device_index = 0
        if not 0 <= device_index < len(self.paths):
            raise ValueError('Invalid device index')
        self.path = self.paths[device_index]
        try:
            for suffix in ('\\h2c_0', '\\c2h_0', '\\user'):
                h = self.k.CreateFileW(self.path + suffix, 0xC0000000, 0,
                                       None, 3, 0x80, None)
                if h == C.c_void_p(-1).value:
                    raise C.WinError(C.get_last_error())
                self.handles.append(h)
            self.buffer = self.k.VirtualAlloc(None, TOTAL_SIZE, 0x3000, 4)
            if not self.buffer:
                raise C.WinError(C.get_last_error())
        except Exception:
            self.close()
            raise

    def enumerate(self):
        from ctypes import wintypes as W

        class GUID(C.Structure):
            _fields_ = [('a', W.DWORD), ('b', W.WORD), ('c', W.WORD), ('d', C.c_ubyte * 8)]

        class Interface(C.Structure):
            _fields_ = [('cbSize', W.DWORD), ('guid', GUID),
                        ('flags', W.DWORD), ('reserved', C.c_size_t)]

        guid = GUID(0x74C7E4A9, 0x6D5D, 0x4A70,
                    (C.c_ubyte * 8)(0xBC, 0x0D, 0x20, 0x69, 0x1D, 0xFF, 0x9E, 0x9D))
        self.s.SetupDiGetClassDevsW.argtypes = [C.POINTER(GUID), W.LPCWSTR, W.HWND, W.DWORD]
        self.s.SetupDiGetClassDevsW.restype = W.HANDLE
        self.s.SetupDiEnumDeviceInterfaces.argtypes = [W.HANDLE, C.c_void_p,
            C.POINTER(GUID), W.DWORD, C.POINTER(Interface)]
        self.s.SetupDiEnumDeviceInterfaces.restype = W.BOOL
        self.s.SetupDiGetDeviceInterfaceDetailW.argtypes = [W.HANDLE,
            C.POINTER(Interface), C.c_void_p, W.DWORD, C.POINTER(W.DWORD), C.c_void_p]
        self.s.SetupDiGetDeviceInterfaceDetailW.restype = W.BOOL
        self.s.SetupDiDestroyDeviceInfoList.argtypes = [W.HANDLE]
        info = self.s.SetupDiGetClassDevsW(C.byref(guid), None, None, 0x12)
        if info == C.c_void_p(-1).value:
            raise C.WinError(C.get_last_error())
        paths = []
        try:
            index = 0
            while True:
                item = Interface()
                item.cbSize = C.sizeof(item)
                if not self.s.SetupDiEnumDeviceInterfaces(info, None, C.byref(guid), index, C.byref(item)):
                    if C.get_last_error() == 259:  # ERROR_NO_MORE_ITEMS
                        break
                    raise C.WinError(C.get_last_error())
                needed = W.DWORD()
                self.s.SetupDiGetDeviceInterfaceDetailW(info, C.byref(item), None, 0, C.byref(needed), None)
                if C.get_last_error() != 122 or needed.value < 6:
                    raise C.WinError(C.get_last_error())
                detail = C.create_string_buffer(needed.value)
                C.cast(detail, C.POINTER(W.DWORD))[0] = 8 if C.sizeof(C.c_void_p) == 8 else 6
                if not self.s.SetupDiGetDeviceInterfaceDetailW(info, C.byref(item), detail,
                                                               needed.value, C.byref(needed), None):
                    raise C.WinError(C.get_last_error())
                paths.append(C.wstring_at(C.addressof(detail) + 4))
                index += 1
        finally:
            self.s.SetupDiDestroyDeviceInfoList(info)
        return paths

    @classmethod
    def list_devices(cls):
        if os.name != 'nt':
            raise RuntimeError('Device enumeration requires Windows')
        obj = cls.__new__(cls)
        obj.s = C.WinDLL('setupapi', use_last_error=True)
        return obj.enumerate()

    def transfer(self, write, address, size, data=None):
        from ctypes import wintypes as W
        check_range(address, size)
        handle = self.handles[0 if write else 1]
        if not self.k.SetFilePointerEx(handle, address, None, 0):
            raise C.WinError(C.get_last_error())
        if write:
            C.memmove(self.buffer, data, size)
        count = W.DWORD()
        fn = self.k.WriteFile if write else self.k.ReadFile
        if not fn(handle, self.buffer, size, C.byref(count), None):
            raise C.WinError(C.get_last_error())
        if count.value != size:
            raise RuntimeError(f'Short DMA at 0x{address:08X}: requested={size} actual={count.value}')
        if not write:
            return C.string_at(self.buffer, size)

    def user_write32(self, address, value):
        data = struct.pack('<I', value & 0xFFFFFFFF)
        self._transfer_handle(self.handles[2], True, address, data)

    def user_read32(self, address):
        return struct.unpack('<I', self._transfer_handle(self.handles[2], False, address, 4))[0]

    def _transfer_handle(self, handle, write, address, data_or_size):
        from ctypes import wintypes as W
        if write:
            data = data_or_size
            size = len(data)
            C.memmove(self.buffer, data, size)
        else:
            size = data_or_size
        if not self.k.SetFilePointerEx(handle, address, None, 0):
            raise C.WinError(C.get_last_error())
        count = W.DWORD()
        fn = self.k.WriteFile if write else self.k.ReadFile
        if not fn(handle, self.buffer, size, C.byref(count), None):
            raise C.WinError(C.get_last_error())
        if count.value != size:
            raise RuntimeError(f'Short user transfer at 0x{address:08X}: requested={size} actual={count.value}')
        return None if write else C.string_at(self.buffer, size)

    def write(self, address, data):
        self.transfer(True, address, len(data), data)

    def read(self, address, size):
        return self.transfer(False, address, size)

    def read_window(self, address, size, window_base, window_size):
        """Read an additional AXI-MM window mapped outside the two-bank range.

        The normal read()/write() methods intentionally guard the validated
        128-KiB dual-bank aperture.  Patch RAM is a separate C2H mapping at
        0x00020000, so it must use this explicit, bounded API instead of
        weakening the source-bank address guard.
        """
        if address < window_base or size <= 0 or address + size > window_base + window_size:
            raise ValueError(f'Mapped-window read outside 0x{window_base:08X}..'
                             f'0x{window_base + window_size - 1:08X}')
        if address % 32 or size % 32:
            raise ValueError('Mapped-window reads must be 32-byte aligned')
        return self._transfer_handle(self.handles[1], False, address, size)

    def close(self):
        for h in self.handles:
            self.k.CloseHandle(h)
        self.handles.clear()
        if self.buffer:
            self.k.VirtualFree(self.buffer, 0, 0x8000)
            self.buffer = None


class MemoryBackend:
    def __init__(self, alias=False):
        self.mem = bytearray(TOTAL_SIZE)
        self.alias = alias

    def write(self, address, data):
        check_range(address, len(data))
        for i, b in enumerate(data):
            a = (address + i) % BANK_SIZE if self.alias else address + i
            self.mem[a] = b

    def read(self, address, size):
        check_range(address, size)
        if self.alias:
            return bytes(self.mem[(address + i) % BANK_SIZE] for i in range(size))
        return bytes(self.mem[address:address + size])


def exercise(io, rounds, report, quiet=False):
    expected = [pattern(0, 0), pattern(1, 0)]
    checked = 0

    def write_bank(bank, data, chunk=BANK_SIZE):
        for offset in range(0, BANK_SIZE, chunk):
            io.write(bank * BANK_SIZE + offset, data[offset:offset + chunk])
        expected[bank] = data

    def verify(name):
        nonlocal checked
        row = {'phase': name, 'banks': []}
        for bank in (0, 1):
            got = io.read(bank * BANK_SIZE, BANK_SIZE)
            want = expected[bank]
            if len(got) != BANK_SIZE:
                raise RuntimeError(f'{name}: short bank read {len(got)}')
            mismatches = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
            if mismatches:
                first = mismatches[0]
                report['failure'] = {'phase': name, 'bank': bank,
                    'mismatch_count': len(mismatches), 'first_offset': first,
                    'axi_address': bank * BANK_SIZE + first,
                    'expected': want[first], 'actual': got[first]}
                raise RuntimeError(f'{name}: bank {bank} errors={len(mismatches)}, '
                    f'address=0x{bank * BANK_SIZE + first:08X} '
                    f'expected={want[first]:02X} got={got[first]:02X}')
            crc = zlib.crc32(got) & 0xFFFFFFFF
            row['banks'].append({'bank': bank, 'bytes': len(got), 'crc32': f'{crc:08X}'})
            checked += len(got)
            if not quiet:
                print(f'PASS {name}: bank={bank} bytes={len(got)} CRC32={crc:08X}', flush=True)
        report.setdefault('phases', []).append(row)

    # Write BOTH before either is read: detects accidental A/B address alias.
    write_bank(0, expected[0])
    write_bank(1, expected[1])
    verify('different_address_patterns')
    for generation in range(1, rounds + 1):
        write_bank(0, pattern(0, generation), TILE_SIZE)
        verify(f'round{generation}_A_only_B_unchanged')
        write_bank(1, pattern(1, generation), TILE_SIZE)
        verify(f'round{generation}_B_only_A_unchanged')
    for name, a, b in [('zero_ones', 0x00, 0xFF), ('checkerboard', 0x55, 0xAA),
                       ('checkerboard_inverse', 0xAA, 0x55)]:
        write_bank(0, bytes([a]) * BANK_SIZE)
        write_bank(1, bytes([b]) * BANK_SIZE)
        verify(name)
    # Restore address-dependent data then overwrite boundary-adjacent AXI beats.
    write_bank(0, pattern(0, rounds + 1))
    write_bank(1, pattern(1, rounds + 1))
    for address in (0, 0x7FE0, 0x8000, 0xFFE0, 0x10000, 0x17FE0, 0x18000, 0x1FFE0):
        data = struct.pack('<8I', *(address ^ (0x13579BDF + i) for i in range(8)))
        io.write(address, data)
        bank, offset = divmod(address, BANK_SIZE)
        exp = bytearray(expected[bank])
        exp[offset:offset + 32] = data
        expected[bank] = bytes(exp)
        verify(f'boundary_write_{address:08X}')
    # Final complete read using tile-size requests, independent of 64-KiB reads.
    for bank in (0, 1):
        got = b''.join(io.read(bank * BANK_SIZE + off, TILE_SIZE)
                       for off in range(0, BANK_SIZE, TILE_SIZE))
        if got != expected[bank]:
            raise RuntimeError(f'Final 4-KiB readback failed in bank {bank}')
        checked += len(got)
    report['bytes_compared'] = checked
    report['result'] = 'PASS'


def self_test():
    r = {}
    exercise(MemoryBackend(), 2, r, quiet=True)
    assert r['result'] == 'PASS'
    try:
        exercise(MemoryBackend(alias=True), 1, {}, quiet=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Bank alias was not detected')
    class CorruptBackend(MemoryBackend):
        def read(self, address, size):
            result = bytearray(super().read(address, size))
            result[19] ^= 1
            return bytes(result)

    class ShortBackend(MemoryBackend):
        def read(self, address, size):
            return super().read(address, size)[:-1]

    for backend in (CorruptBackend(), ShortBackend()):
        try:
            exercise(backend, 1, {}, quiet=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Corruption/short read was not detected')
    for address, size in [(0x20000, 32), (-32, 32), (0x1FFE0, 64), (1, 32)]:
        try:
            check_range(address, size)
        except ValueError:
            continue
        raise AssertionError('Range/alignment guard failed')
    print('SELF-TEST PASS: patterns, byte comparison, bank alias, corruption, short read, address guards; NO hardware accessed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--list', action='store_true')
    mode.add_argument('--run', action='store_true', help='Overwrite both test BRAM banks and verify')
    parser.add_argument('--device-index', type=int)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.list:
        for i, path in enumerate(WindowsXDMA.list_devices()):
            print(i, path)
        return 0
    if not 1 <= args.rounds <= 10000:
        parser.error('--rounds must be 1..10000')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = args.report or Path(__file__).resolve().parent / f'bank_test_{stamp}.json'
    if out.exists():
        raise FileExistsError(f'Report exists; choose a new path: {out}')
    report = {'mode': 'hardware', 'start': stamp, 'rounds': args.rounds,
              'result': 'FAIL', 'window': '0x00000000..0x0001FFFF'}
    io = None
    start = time.perf_counter()
    try:
        io = WindowsXDMA(args.device_index)
        report['device'] = io.path
        print('Testing two BRAM banks; BOTH contents will be overwritten.', flush=True)
        exercise(io, args.rounds, report)
        print('PASS: PCIe -> two independent 64-KiB banks -> full byte-exact readback.', flush=True)
        return 0
    except Exception as e:
        report['error'] = str(e)
        print('FAIL:', e, file=sys.stderr, flush=True)
        return 1
    finally:
        if io:
            io.close()
        report['elapsed_seconds'] = time.perf_counter() - start
        out.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Report:', out, flush=True)


if __name__ == '__main__':
    sys.exit(main())
