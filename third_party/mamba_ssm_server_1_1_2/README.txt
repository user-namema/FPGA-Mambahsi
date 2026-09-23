# Mamba 1.1.2 — observed server Python snapshot

Received 2026-09-23. Fourteen Python source files are preserved byte-for-byte.
The license and AUTHORS are in `distribution_metadata/`.

**Modification notice:** `mamba_ssm/modules/mamba_simple.py` differs from the
installed wheel RECORD. It was already modified on the supplied server; this
repository preserves that version. The import audit records both hashes.
`selective_scan_interface.py`, `mamba_simple.py` and `__init__.py` match the
server environment collector hashes. RECORD describes the original wheel,
not the later modified file. The received Python snapshot has no subsequent
packaging edits.

This directory is not an installable wheel. It excludes `.so`, `.pyc` and cache
files. The recorded interface imports both `selective_scan_cuda` and
`causal_conv1d_cuda`; compatible binaries are required. Follow
`../../docs/ENVIRONMENT.txt` for explicit `PYTHONPATH` selection on the existing
server, and run the CUDA forward/backward check before replay.
