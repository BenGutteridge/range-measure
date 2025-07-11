from functools import wraps
import torch
import gc
from os.path import join
from loguru import logger


class GPUTracker:

    def __init__(self, device):
        self.device = device

    def start(self):
        self.initial_memory = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_max_memory_allocated(self.device)

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        self.start_event.record()

    def end(self, idx: int, logpath: str):
        """
        idx: batch idx
        logpath: path to log file
        """
        self.end_event.record()
        torch.cuda.synchronize(self.device)
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

        final_memory = torch.cuda.memory_allocated(self.device)
        peak_memory = torch.cuda.max_memory_allocated(self.device)

        # Compute net and peak memory usage in gigabytes.
        net_memory_used_gb = (final_memory - self.initial_memory) / (1024**3)
        peak_memory_used_gb = peak_memory / (1024**3)

        # Append a new line to csv file: columns are idx, elapsed time in ms, net & max memory in gb
        with open(logpath, "a") as f:
            f.write(
                f"{idx},{elapsed_time_ms},{net_memory_used_gb},{peak_memory_used_gb}\n"
            )
        logger.info(
            f"Batch {idx} done.\nElapsed time: {elapsed_time_ms:.1f} ms.\nNet memory used: {net_memory_used_gb:.2f} GB.\nPeak memory used: {peak_memory_used_gb:.2f} GB.\nSaved to {logpath}."
        )
