"""Copy an XSim SAIF while renaming only malformed '(null)' NET identifiers.

The source is never modified. --scan-only identifies anomalous lines without
writing an 18+ GB copy. A JSON manifest binds any output to its exact source.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

CHUNK = 32 * 1024 * 1024
METHOD = "xsim-null-net-name-v1"


def source_hash_from_capture(receipt):
    for line in receipt.read_text(encoding="utf-8").splitlines():
        if line.startswith("set capture_saif_hash "):
            return line.split(" ", 2)[2].strip("{} ")
    raise ValueError("capture_saif_hash missing from receipt")


def process(source, output, expected_hash, scan_only):
    source = source.resolve(strict=True)
    if not scan_only:
        output = output.resolve()
        if output == source or output.exists():
            raise ValueError("Output must be a NEW file separate from source")
        partial = Path(str(output) + ".partial")
        manifest = Path(str(output) + ".json")
        if partial.exists() or manifest.exists():
            raise ValueError("Partial or manifest already exists; choose a new output name")
        free = shutil.disk_usage(output.parent).free
        if free < source.stat().st_size + 5 * 1024**3:
            raise ValueError("Output drive needs source size plus 5 GiB free space")
    original_hash = hashlib.sha256()
    clean_hash = hashlib.sha256() if not scan_only else None
    renamed = 0
    first_line = last_line = 0
    lines_seen = 0
    nul_count = 0
    original_bytes = 0
    clean_bytes = 0
    examples = []
    writer = None if scan_only else partial.open("xb", buffering=4 * 1024 * 1024)
    try:
        with source.open("rb", buffering=4 * 1024 * 1024) as reader:
            carry = b""
            while True:
                raw = reader.read(CHUNK)
                if not raw:
                    break
                original_hash.update(raw)
                original_bytes += len(raw)
                nul_count += raw.count(b"\x00")
                joined = carry + raw
                end = joined.rfind(b"\n")
                if end < 0:
                    carry = joined
                    if len(carry) > 1024 * 1024:
                        raise ValueError("Unusually long SAIF line")
                    continue
                complete, carry = joined[:end + 1], joined[end + 1:]
                if b"null" not in complete:
                    lines_seen += complete.count(b"\n")
                    if writer:
                        writer.write(complete)
                        clean_hash.update(complete)
                        clean_bytes += len(complete)
                    continue
                for line in complete.splitlines(keepends=True):
                    lines_seen += 1
                    cleaned = line
                    if b"null" in line:
                        front, separator, suffix = line.partition(b" (T0 ")
                        stripped = front.lstrip(b" \t")
                        if not separator or not stripped.startswith(b"(") or b"null" not in stripped:
                            raise ValueError(f"Unsupported null occurrence on line {lines_seen}: {line[:160]!r}")
                        name = stripped[1:]
                        safe = b"XsimNull_" + hashlib.blake2b(name, digest_size=16).hexdigest().encode("ascii")
                        prefix = front[:len(front) - len(stripped)]
                        cleaned = prefix + b"(" + safe + separator + suffix
                        renamed += 1
                        first_line = first_line or lines_seen
                        last_line = lines_seen
                        if len(examples) < 5:
                            examples.append({"line": lines_seen, "original_name": name.decode("ascii", "replace"),
                                             "replacement_name": safe.decode("ascii")})
                    if writer:
                        writer.write(cleaned)
                        clean_hash.update(cleaned)
                        clean_bytes += len(cleaned)
                if original_bytes // (2 * 1024**3) != (original_bytes - len(raw)) // (2 * 1024**3):
                    print(f"SCANNED {original_bytes / 1024**3:.1f} GiB, renamed={renamed}", flush=True)
            if carry:
                raise ValueError(f"SAIF lacks final newline; trailing bytes={len(carry)}")
        if writer:
            writer.flush()
            os.fsync(writer.fileno())
    finally:
        if writer:
            writer.close()
    original_digest = original_hash.hexdigest()
    if expected_hash and original_digest != expected_hash:
        raise ValueError(f"Source SHA256 differs from capture receipt: {original_digest}")
    if nul_count:
        raise ValueError(f"Source contains {nul_count} NUL bytes; targeted name repair is insufficient")
    if renamed == 0:
        raise ValueError("No null NET names found; no change to make")
    result = {
        "method": METHOD,
        "source": str(source),
        "source_sha256": original_digest,
        "source_bytes": original_bytes,
        "output": str(output) if output else None,
        "output_sha256": clean_hash.hexdigest() if writer else None,
        "output_bytes": clean_bytes if writer else None,
        "renamed_names": renamed,
        "first_line": first_line,
        "last_line": last_line,
        "total_lines": lines_seen,
        "nul_bytes": nul_count,
        "examples": examples,
        "power_coverage_review": "REQUIRED: renamed names are expected to remain unmatched",
    }
    if writer:
        if partial.stat().st_size != clean_bytes:
            raise ValueError("Output byte count mismatch")
        os.replace(partial, output)
        with manifest.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--capture-receipt", type=Path)
    ap.add_argument("--scan-only", action="store_true")
    args = ap.parse_args()
    if not args.scan_only and not args.output:
        ap.error("--output is required unless --scan-only is given")
    if args.capture_receipt and not args.capture_receipt.is_file():
        ap.error("Capture receipt not found")
    expected_hash = source_hash_from_capture(args.capture_receipt) if args.capture_receipt else None
    process(args.source, args.output, expected_hash, args.scan_only)


if __name__ == "__main__":
    main()
