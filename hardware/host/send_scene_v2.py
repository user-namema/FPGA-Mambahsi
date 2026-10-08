"""E2 ABI: batch bank input, actual II statistics, actual DDR labels via C2H.

Input bytes are the exact exported UINT8 Patch tile bytes. No re-quantization.
Source writes overwrite 0..0x1ffff; readback uses 0x20000..0x20fff only.
Only run with the NEW E2 bitstream and a reset, otherwise refuse operation.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'host'))
from dual_bank_test import WindowsXDMA

TILES, TILE_BYTES, PITCH, HEIGHT, WIDTH = 858, 4096, 384, 610, 340
PALETTE = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
           (255, 0, 255), (0, 255, 255), (200, 100, 0), (0, 200, 100), (100, 0, 200))


def basic(io):
    s = io.user_read32(8)
    if (s >> 8) & 255 != 0xE2:
        raise RuntimeError(f'Wrong bitstream: status {s:08X}, expected E2')
    if s & 8:
        raise RuntimeError(f'FPGA protocol error: {s:08X}')
    return s


def command(io, value):
    io.user_write32(0, 0)
    io.user_write32(0, value | 1)
    io.user_write32(0, 0)


def wait(io, condition, label, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        s = basic(io)
        if condition(s):
            return s
        # No fixed 1 ms sleep: a bank covers only 655.36 us at 100 MHz.
    raise TimeoutError(f'{label}, last status={s:08X}')


def query(io, page):
    io.user_write32(0, page << 28)
    value = io.user_read32(8)
    io.user_write32(0, 0)
    return value


def read_labels(io):
    pages = []
    for page in range((PITCH * HEIGHT + 4095) // 4096):
        command(io, 6 | (page << 8))
        wait(io, lambda s: bool(s & 64) and not s & 128, f'DDR page {page}')
        if query(io, 5) != page:
            raise RuntimeError('Readback page acknowledgement mismatch')
        pages.append(io.read_window(0x20000, 4096, 0x20000, 0x1000))
    padded = b''.join(pages)[:PITCH * HEIGHT]
    labels = b''.join(padded[y*PITCH:y*PITCH+WIDTH] for y in range(HEIGHT))
    if max(labels) > 8:
        raise RuntimeError('Returned DDR contains an invalid class ID (>8)')
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', required=True, type=Path)
    ap.add_argument('--expected-labels', type=Path, help='610x340 uint8, IDs 0..8 (.bin or .npy)')
    ap.add_argument('--verify-source', action='store_true', help='diagnostic mode; extra C2H may starve input')
    ap.add_argument('--require-ii4096', action='store_true', help='fail if any start interval exceeds 4096')
    ap.add_argument('--device-index', type=int)
    ap.add_argument('--output', type=Path, default=ROOT / ('result_' + str(time.time_ns())))
    args = ap.parse_args()
    report = {'result': 'FAIL', 'abi': 'E2', 'clock_cycles_per_tile_target': 4096}
    io = None
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        raw = args.scene.read_bytes()
        if len(raw) != TILES * TILE_BYTES:
            raise ValueError(f'Expected {TILES*TILE_BYTES} raw tile bytes, got {len(raw)}')
        report['input_sha256'] = hashlib.sha256(raw).hexdigest()
        io = WindowsXDMA(args.device_index)
        io.user_write32(0, 0)
        initial = basic(io)
        if initial & 0xF6 or initial >> 16:
            raise RuntimeError('Scene or page already active; reset board first')
        wait(io, lambda s: bool(s & 1), 'DDR/network/HDMI initialization')
        started = time.monotonic()
        for batch, first in enumerate(range(0, TILES, 16)):
            bank = batch & 1
            count = min(16, TILES-first)
            wait(io, lambda s: not s & (2 << bank), f'ownership release bank {bank}')
            payload = raw[first*TILE_BYTES:(first+count)*TILE_BYTES]
            io.write(bank * 65536, payload)
            if args.verify_source and io.read(bank * 65536, len(payload)) != payload:
                raise RuntimeError(f'Input readback mismatch at tile {first}')
            command(io, 2 | (bank << 4) | (count << 5))
            if query(io, 6) != first + count:
                raise RuntimeError('Bank commit acknowledgement mismatch')
            # Preload BOTH banks before GO; no per-tile host commands.
            if batch == 1:
                command(io, 4)
        wait(io, lambda s: s >> 16 == TILES and bool(s & 16), 'full DDR frame', 120)
        report['transfer_and_inference_seconds'] = time.monotonic() - started
        time.sleep(0.001)  # statistic words now frozen in source domain
        report.update(min_ii=query(io, 1), max_ii=query(io, 2),
                      non4096_intervals=query(io, 3), tiles_started=query(io, 4))
        diagnostics = query(io, 7)
        if not diagnostics & 2:
            raise RuntimeError('Statistics snapshot is not ready')
        report['display_underflow'] = bool(diagnostics & 1)
        labels = read_labels(io)
        report['display_underflow'] = report['display_underflow'] or bool(query(io, 7) & 1)
        (args.output / 'labels_u8.bin').write_bytes(labels)
        rgb = bytes(v for label in labels for v in PALETTE[label])
        (args.output / 'prediction.ppm').write_bytes(f'P6\n{WIDTH} {HEIGHT}\n255\n'.encode() + rgb)
        report['labels_sha256'] = hashlib.sha256(labels).hexdigest()
        report['labels_count'] = len(labels)
        report['label_reference_checked'] = args.expected_labels is not None
        if args.expected_labels:
            if args.expected_labels.suffix.lower() == '.npy':
                import numpy as np
                ref_array = np.load(args.expected_labels, allow_pickle=False)
                if ref_array.shape != (HEIGHT, WIDTH) or not np.all((ref_array >= 0) & (ref_array <= 8)):
                    raise ValueError('Reference must have shape (610,340), IDs 0..8')
                ref = ref_array.astype(np.uint8).tobytes()
            else:
                ref = args.expected_labels.read_bytes()
            if len(ref) != len(labels):
                raise ValueError('Reference byte count mismatch')
            mismatch = [i for i, (a, b) in enumerate(zip(labels, ref)) if a != b]
            report['label_mismatches'] = len(mismatch)
            report['first_mismatch_yx'] = divmod(mismatch[0], WIDTH) if mismatch else None
            if mismatch:
                raise RuntimeError(f'{len(mismatch)} DDR label mismatches')
        exact_ii = report['min_ii'] == report['max_ii'] == 4096 and report['non4096_intervals'] == 0
        report['ii4096_pass'] = exact_ii
        if args.require_ii4096 and not exact_ii:
            raise RuntimeError('Actual II not continuously 4096; inspect host starvation and RTL statistics')
        report['result'] = 'PASS' if args.expected_labels else 'READBACK_OK_REFERENCE_NOT_CHECKED'
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        report['error'] = str(exc)
        print('FAIL:', exc, file=sys.stderr)
        return 1
    finally:
        if io is not None:
            io.close()
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
