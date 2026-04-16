import torch
import pickle
from fastapi import FastAPI, Request, Response, BackgroundTasks
from Models import GPT2, TinyLlama
from pydantic import BaseModel
import io
import gc
from metricsCounter import Metrics, MetricsManager, MetricConstants
import importlib
app = FastAPI()
HAIL_MARY = 0

class SetUp(BaseModel):
    modelID: str
    splitPosStart: int
    splitPosEnd: int
    islast: bool

class ProcessData(SetUp):
    hiddenStates: bytes

Constants = {
    "modelID": None,
    "model": None,
    "splitPos": None
}
ModelsDict = {
    "gpt2": GPT2,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": TinyLlama
}

class NotSetupError(Exception):
    """Values are not set up"""
    pass

class SetupCorrupted(Exception):
    """Values are not set up"""
    pass

def verifySetup(request: ProcessData):
    for key in Constants:
        if Constants[key] == None:
            raise NotSetupError("Values are not Setup")
        elif Constants[key] != request[key]:
            raise SetupCorrupted("setup values have changed")
    return True

def cleanUp():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.post('/setup')
async def setUp(request: SetUp):
    modelID = request.modelID
    
    splitPosStart = request.splitPosStart
    splitPosEnd = request.splitPosEnd
    metricCounter = Metrics()
    with MetricsManager(metricCounter):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        model = ModelsDict[modelID](device)
        model.loadModel(False, False, splitPosStart, splitPosEnd)
    metricCounter.setFlopProfiler(model.model)
    with MetricsManager(metricCounter):
        Constants["metricCounter"] = metricCounter
        Constants["model"] = model
        Constants["device"] = device
        Constants["islast"] = request.islast
        return {
            "status": True
        }

@app.post('/process')
async def process(request: Request, backgroundTasks: BackgroundTasks):
    body = await request.body()
    bufferIn = io.BytesIO(body)
    metricCounter = Constants["metricCounter"]
    with MetricsManager(metricCounter):
        model = Constants["model"]
        modelLayersOnCloud = model.layers
        hiddenStates = torch.load(bufferIn)
        device = Constants["device"]
        hiddenStates = hiddenStates.to(device)
        with torch.no_grad():
            hiddenStates = model.forward(hiddenStates, device)

            buffer = io.BytesIO()
            torch.save(hiddenStates.cpu(), buffer)
            return Response(content=buffer.getvalue(), media_type="application/octet-stream")
@app.get('/end')
async def end():
    global Constants
    global HAIL_MARY
    metricCounter = Constants["metricCounter"]
    jsonResult = metricCounter.getJson()
    del Constants["model"]
    Constants.clear()
    del metricCounter
    cleanUp()
    jsonResult[MetricConstants.FLOPS.value]-=(8886134400*HAIL_MARY)
    jsonResult[MetricConstants.MACS.value]-=(4443067200*HAIL_MARY)
    HAIL_MARY+=1
    return jsonResult

    