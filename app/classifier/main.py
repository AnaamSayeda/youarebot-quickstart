from fastapi import FastAPI
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
    # Dummy probability generator
    prob = random.uniform(0, 1)
    return {"is_bot_probability": prob}
