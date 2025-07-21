import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel


app = FastAPI()

@app.post("/get_message")
async def get_message(request: Request):
    body = await request.json()
    prompt = body["prompt"]

    # Send to LLM server
    res = requests.post("http://llm:11434/api/generate", json={
        "model": "orca-mini",
        "prompt": prompt,
        "stream": False
    })

    reply = res.json()["response"]
    return {"response": reply}
class PredictionRequest(BaseModel):
    id: str
    dialog_id: str
    participant_index: int
    text: str

class PredictionResponse(BaseModel):
    is_bot_probability: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Dummy logic: just generate a random probability
    prob = random.uniform(0.0, 1.0)
    return {"is_bot_probability": prob}
