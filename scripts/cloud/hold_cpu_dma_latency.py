"""Holds /dev/cpu_dma_latency at 0 for as long as this process runs (the scorer VM's life; scorer.sh starts it).

The scorer VM's guest offers a C2 idle state with 800 us exit latency; holding the PM QoS CPU latency limit at 0
keeps idle cores out of it, so a short multi-threaded solver call does not pay the wake-up (scoring investigation,
2026-10-10). The kernel keeps the limit while the file stays open and drops it when this process exits.
Usage (root): python3 hold_cpu_dma_latency.py [device]
"""

import os
import signal
import struct
import sys
import time

path = sys.argv[1] if len(sys.argv) > 1 else "/dev/cpu_dma_latency"
fd = os.open(path, os.O_RDWR)
os.write(fd, struct.pack("i", 0))
print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} holding {path} at 0 (pid {os.getpid()})", flush=True)
while True:
    signal.pause()
