from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (relative path fix)
model_path = "models/gemma-3-1b-it-Q4_K_M.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)

@app.get("/")
def read_root():
    return {"message": "LLM backend is running 🧠⚡"}

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
