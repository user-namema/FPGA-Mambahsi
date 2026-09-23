"""Compatibility entry point. Active implementation: both_FPGA_single_qat_source.py."""
from both_FPGA_single_qat_source import *  # noqa: F401,F403
import both_FPGA_single_qat_source as _source

def __getattr__(name):
    return getattr(_source, name)

if __name__ == "__main__":
    main()
