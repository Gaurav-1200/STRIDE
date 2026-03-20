import torch
import pickle
from fastapi import FastAPI, Request, Response, BackgroundTasks
from Models import GPT2, TinyLlama
from pydantic import BaseModel
import io

app = FastAPI()


class SetUp(BaseModel):
    modelID: str
    splitPosStart: int
    splitPosEnd: int

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.post('/setup')
async def setUp(request: SetUp):
    modelID = request.modelID
    splitPosStart = request.splitPosStart
    splitPosEnd = request.splitPosEnd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model = ModelsDict[modelID](device)
    model.loadModel(False, False, splitPosStart, splitPosEnd)
    Constants["model"] = model
    Constants["device"] = device
    return {
        "status": True
    }

@app.post('/process')
async def process(request: Request, backgroundTasks: BackgroundTasks):
    body = await request.body()
    bufferIn = io.BytesIO(body)
    model = Constants["model"]
    # model.loadModel(False, False, *Constants["splitPos"]) 
    modelLayersOnCloud = model.layers
    hiddenStateFromLocal = torch.load(bufferIn)
    hiddenStates = hiddenStateFromLocal
    seq_len = hiddenStates.shape[1] 
    device = Constants["device"]
    hiddenStates = hiddenStates.to(device)
    # device = "cpu"
    # 1. Regenerate Position IDs locally on the cloud GPU
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
    position_embeddings = model.model.model.rotary_emb(hiddenStates, position_ids.to(device))
    attention_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len),device=device))
    attention_mask = (1.0 - attention_mask) * torch.finfo(hiddenStates.dtype).min
    attention_mask = attention_mask.to(hiddenStates.dtype)
    with torch.no_grad():
        for layer in modelLayersOnCloud:
            hiddenStates = layer(hiddenStates,attention_mask=attention_mask, position_ids=position_ids,position_embeddings=position_embeddings)

        buffer = io.BytesIO()
        torch.save(hiddenStates.cpu(), buffer)
        backgroundTasks.add_task(cleanUp)
        return Response(content=buffer.getvalue(), media_type="application/octet-stream")

    