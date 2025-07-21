from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 轻量多语言模型：适合 Render 免费计划部署
MODEL_NAME = "Jayveersinh-Raj/PolyGuard"
DEVICE = 0 if os.getenv("USE_CUDA", "false") == "true" else -1

print(f"Loading model: {MODEL_NAME} on {'cuda' if DEVICE == 0 else 'cpu'}")

# Load model with optimizations
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    torchscript=True,  # Optimize with TorchScript
    low_cpu_mem_usage=True  # Reduce memory usage during loading
)

# Move to appropriate device
device = torch.device("cuda" if DEVICE == 0 else "cpu")
model.to(device)

class TextModerationRequest(BaseModel):
    input: str

@app.post("/text-moderate")
async def moderate_text(req: TextModerationRequest):
    # Use tokenizer and model directly instead of pipeline
    inputs = tokenizer(req.input, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get prediction
    predicted_class = outputs.logits.argmax().item()
    score = torch.softmax(outputs.logits, dim=1)[0][predicted_class].item()
    
    # Map to output format - adjusted to match README format
    label = "not_offensive" if predicted_class == 0 else "offensive"
    
    return {
        "result": [
            {
                "label": label,
                "score": score
            }
        ]
    }

@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME}
