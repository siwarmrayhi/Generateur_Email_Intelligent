from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

app = FastAPI(title="Email AI Generator API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle au démarrage
print(" Chargement du modèle...")
MODEL_PATH = "../models/saved_model"

try:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f" Modèle chargé sur {device}")
except Exception as e:
    print(f" Erreur de chargement: {e}")
    tokenizer = None
    model = None

class EmailRequest(BaseModel):
    context: str
    tone: str = "neutre"
    max_length: int = 300

class EmailResponse(BaseModel):
    email: str
    context: str
    tone: str

@app.get("/")
def root():
    return {
        "message": "Email AI Generator API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.post("/generate", response_model=EmailResponse)
def generate_email(request: EmailRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    # Créer le prompt
    prompt = f"[CONTEXT] {request.context} [TONE] {request.tone} [EMAIL]"
    
    # Tokenize
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Générer
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=request.max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Décoder
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraire seulement l'email (après [EMAIL])
    if "[EMAIL]" in generated_text:
        email_content = generated_text.split("[EMAIL]")[-1].strip()
    else:
        email_content = generated_text
    
    return EmailResponse(
        email=email_content,
        context=request.context,
        tone=request.tone
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "loaded" if model else "not loaded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
