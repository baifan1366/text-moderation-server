from fastapi import FastAPI, HTTPException
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
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        torchscript=True,  # Optimize with TorchScript
        low_cpu_mem_usage=True  # Reduce memory usage during loading
    )

    # Move to appropriate device
    device = torch.device("cuda" if DEVICE == 0 else "cpu")
    model.to(device)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Continue execution, but the endpoints will handle the error

class TextModerationRequest(BaseModel):
    input: str

@app.post("/text-moderate")
async def moderate_text(req: TextModerationRequest):
    try:
        # Check if model was loaded successfully
        if 'model' not in globals() or 'tokenizer' not in globals():
            raise HTTPException(status_code=500, detail="Model or tokenizer not loaded properly")
        
        # Check for empty input
        if not req.input or len(req.input.strip()) == 0:
            return {
                "result": [
                    {
                        "label": "not_offensive",
                        "score": 1.0
                    }
                ]
            }
            
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
    except Exception as e:
        print(f"Error in text moderation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@app.get("/")
def health():
    try:
        # Check if model was loaded successfully
        if 'model' not in globals() or 'tokenizer' not in globals():
            return {
                "status": "warning",
                "model": MODEL_NAME,
                "message": "Model not loaded properly"
            }
        return {"status": "ok", "model": MODEL_NAME}
    except Exception as e:
        return {
            "status": "error",
            "model": MODEL_NAME,
            "message": str(e)
        }
