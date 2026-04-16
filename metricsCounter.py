import time
from deepspeed.profiling.flops_profiler import FlopsProfiler
import json
from enum import Enum
import torch
from deepspeed.profiling.flops_profiler import profiler as ds_profiler
class MetricConstants(Enum):
    MACS = "MACS"
    FLOPS = "FLOPS"
    PEAKVRAM = "Peak VRAM"
    TimeTaken = "Time"

# if not getattr(ds_profiler, "_ds_patched_already", False):
#     ds_profiler._ds_patched_already = True

#     _orig_patch_functionals = ds_profiler._patch_functionals
#     def safe_patch_functionals():
#         if not getattr(ds_profiler, "_functionals_patched", False):
#             _orig_patch_functionals()
#             ds_profiler._functionals_patched = True
#     ds_profiler._patch_functionals = safe_patch_functionals

#     _orig_patch_tensor_methods = ds_profiler._patch_tensor_methods
#     def safe_patch_tensor_methods():
#         if not getattr(ds_profiler, "_tensor_patched", False):
#             _orig_patch_tensor_methods()
#             ds_profiler._tensor_patched = True
#     ds_profiler._patch_tensor_methods = safe_patch_tensor_methods

#     _orig_patch_misc = ds_profiler._patch_miscellaneous_operations
#     def safe_patch_misc():
#         if not getattr(ds_profiler, "_misc_patched", False):
#             _orig_patch_misc()
#             ds_profiler._misc_patched = True
#     ds_profiler._patch_miscellaneous_operations = safe_patch_misc
class Metrics:
    def __init__(self, model=None):
        self.flopProfiler = FlopsProfiler(model) if model!=None else None
        self.flopProfiler = None
        self.flops = 0
        self.macs = 0
        self.totalTime = 0
        self.peakVramUsage = 0
        self.startTime = None
    def setFlopProfiler(self, model):
        self.flopProfiler = FlopsProfiler(model)
        
    def start(self):
        self.startTime = time.perf_counter()
        if self.flopProfiler is not None:
            torch.cuda.reset_peak_memory_stats()
            self.flopProfiler.start_profile()
            self.flopProfiler.reset_profile()
    def end(self):
        if self.flopProfiler is not None:
            self.flopProfiler.stop_profile()
            self.peakVramUsage = max(self.peakVramUsage, torch.cuda.max_memory_allocated()/(1024**2))
            self.flops += self.flopProfiler.get_total_flops()
            self.macs += self.flopProfiler.get_total_macs()
            self.flopProfiler.end_profile()
        endTime = time.perf_counter()
        self.totalTime += (endTime - self.startTime)
    def getJson(self):
        return {
            MetricConstants.FLOPS.value: self.flops,
            MetricConstants.MACS.value: self.macs,
            MetricConstants.PEAKVRAM.value: self.peakVramUsage,
            MetricConstants.TimeTaken.value: self.totalTime
        }
    @staticmethod
    def saveUsageDict(*metrics, metricPath):
        metrics = [metric for metric in metrics if metric is not None]
        deviceNames = ["Local", "Server"]
        deviceCount = len(metrics)
        metricsDict = {}
        for idx in range(deviceCount):
            device = deviceNames[idx]
            metricsDict[device] = {}
            for metricToMeasure in MetricConstants:
                value = metricToMeasure.value
                metricsJson  = metrics[idx]
                # metricsDict[f"{value} ({deviceNames[idx]})"] = metricsJson[value]
                metricsDict[device][value] = metricsJson[value]
        with open(metricPath,"w") as f:
            f.write(json.dumps(metricsDict, indent=2))

class MetricsManager:
    def __init__(self, metrics):
        self.metrics = metrics
    def __enter__(self):
        self.metrics.start()
    def __exit__(self, exc_type, exc, tb):
        self.metrics.end()
