import time
from deepspeed.profiling.flops_profiler import FlopsProfiler
import json
from enum import Enum
import torch
class MetricConstants(Enum):
    MACS = "MACS"
    FLOPS = "FLOPS"
    PEAKVRAM = "Peak VRAM"
    TimeTaken = "Time"


class Metrics:
    def __init__(self, model=None):
        self.flopProfiler = FlopsProfiler(model) if model!=None else None 
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
    def end(self):
        if self.flopProfiler is not None:
            self.flopProfiler.stop_profile()
            self.peakVramUsage = max(self.peakVramUsage, torch.cuda.max_memory_allocated()/(1024**2))
            self.flops += self.flopProfiler.get_total_flops()
            self.macs += self.flopProfiler.get_total_macs()
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
        metrics = [metric for metric in metrics if metric!=None]
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
