from transformers import pipeline
import mlflow.pyfunc

# Load model from MLflow registry
from mlflow.transformers import load_model

model_uri = "models:/bot-human-classifier/Production"  # Or change to latest run URI
zero_shot = pipeline("text-classification", model=load_model(model_uri))
