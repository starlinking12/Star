"""
FastAPI server for OpenMythos inference.
Provides /generate, /scan_vulnerability, and /health endpoints.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))

from open_mythos import OpenMythos, MythosConfig


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0, le=200)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    do_sample: bool = True
    n_loops: int = Field(default=32, ge=1, le=128)


class GenerateResponse(BaseModel):
    text: str
    generated_tokens: int


class ScanRequest(BaseModel):
    code: str
    language: str = "c"
    n_loops: int = Field(default=32, ge=1, le=128)


class Vulnerability(BaseModel):
    line: int
    type: str
    severity: str
    description: str


class ScanResponse(BaseModel):
    vulnerabilities: List[Vulnerability]
    safe: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ============================================================================
# Global Variables
# ============================================================================

app = FastAPI(title="Mythos API", version="1.0.0")
model = None
device = "cpu"


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup():
    global model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MythosConfig()
    model = OpenMythos(cfg)
    
    # Try to load pretrained weights if they exist
    checkpoint_path = Path("checkpoints/best_model.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")
    
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if model is not None else "loading",
        model_loaded=model is not None,
        device=device
    )


# ============================================================================
# Generate Endpoint
# ============================================================================

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Simple tokenization (replace with proper tokenizer in production)
    input_ids = torch.tensor([[ord(c) % model.cfg.vocab_size for c in request.prompt[:1000]]])
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            n_loops=request.n_loops,
        )
    
    # Simple decoding (replace with proper tokenizer in production)
    output_text = ''.join(chr(id % 128) for id in output_ids[0].tolist() if id < 128)
    generated_tokens = output_ids.shape[1] - prompt_len
    
    return GenerateResponse(text=output_text, generated_tokens=generated_tokens)


# ============================================================================
# Scan Vulnerability Endpoint
# ============================================================================

@app.post("/scan_vulnerability", response_model=ScanResponse)
async def scan_vulnerability(request: ScanRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Build prompt for vulnerability analysis
    prompt = f"""Analyze the following {request.language} code for security vulnerabilities:

```{request.language}
{request.code}