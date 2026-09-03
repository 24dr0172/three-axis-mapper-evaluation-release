"""Resource Monitor for Wall-Clock, CPU Time, and Memory Tracking."""

import resource
import time
from typing import Dict, Optional


class ResourceMonitor:
    """Context manager for accurate wall-time, CPU-time, and peak memory tracking."""

    def __init__(self):
        self.wall_time_seconds: float = 0.0
        self.cpu_time_seconds: float = 0.0
        self.peak_memory_bytes: Optional[int] = None
        self._start_wall: float = 0.0
        self._start_cpu: float = 0.0

    def __enter__(self) -> "ResourceMonitor":
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        self.wall_time_seconds = end_wall - self._start_wall
        self.cpu_time_seconds = end_cpu - self._start_cpu

        try:
            # ru_maxrss in kilobytes on Linux
            usage = resource.getrusage(resource.RUSAGE_SELF)
            self.peak_memory_bytes = int(usage.ru_maxrss * 1024)
        except Exception:
            self.peak_memory_bytes = None

    def get_summary(self) -> Dict[str, Optional[float]]:
        return {
            "wall_time_seconds": self.wall_time_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
        }
