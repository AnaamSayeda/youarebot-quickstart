from fastapi import FastAPI, Request
import requests
from pydantic import BaseModel
import random

app = FastAPI()

class PredictionRequest(BaseModel):
    id: str
    dialog_id: str
    participant_index: int
    text: str

class PredictionResponse(BaseModel):
    is_bot_probability: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    res = requests.post("http://classifier:8000/predict", json=request.dict())
    return res.json()

@app.post("/get_message")
async def get_message(request: Request):
    body = await request.json()
    res = requests.post("http://llm:11434/v1/chat/completions", json={
        "model": "llama3",
        "messages": [{"role": "user", "content": body["last_msg_text"]}],
        "stream": False
    })
    reply = res.json()["choices"][0]["message"]["content"]
    return {
        "choices": [
            {"message": {"content": reply}}
        ]
    }

