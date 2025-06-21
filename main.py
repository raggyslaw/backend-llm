from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import os

app = FastAPI()

# CORS setup to allow requests from your browser extension or frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your locally hosted GGUF model
model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "models/gemma-3-1b-it-Q4_K_M.gguf")
)

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)

# Basic test route
@app.get("/")
def read_root():
    return {"message": "LLM backend is running 🧠⚡"}

# Main LLM endpoint
@app.post("/generate")
async def generate_response(_: Request):
    fixed_instruction = (
        "Write JavaScript that selects all visible <input>, <textarea>, and contenteditable elements "
        "on the page, and moves them to the top of the <body> inside a floating container. "
        "Keep their original styles intact, and hide their original parents to avoid duplication. "
        "The layout must look clean and not overlap existing UI."
    )

    response = llm(
        prompt=f"[INST] {fixed_instruction} [/INST]",
        max_tokens=512,
        stop=["</s>"]
    )

    return {"response": response["choices"][0]["text"].strip()}
