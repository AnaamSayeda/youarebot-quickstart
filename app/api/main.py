from fastapi import FastAPI
from pydantic import BaseModel
from uuid import UUID, uuid4
from transformers import pipeline

app = FastAPI()

# Load the zero-shot model once at startup
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

class PredictionRequest(BaseModel):
    text: str
    dialog_id: UUID
    id: UUID
    participant_index: int

class PredictionResponse(BaseModel):
    id: UUID
    message_id: UUID
    dialog_id: UUID
    participant_index: int
    is_bot_probability: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    candidate_labels = ["bot", "human"]
    result = classifier(request.text, candidate_labels)
    bot_score = result["scores"][result["labels"].index("bot")]

    return PredictionResponse(
        id=request.id,
        message_id=uuid4(),
        dialog_id=request.dialog_id,
        participant_index=request.participant_index,
        is_bot_probability=bot_score
    )
