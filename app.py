import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import os  # <- port ke liye

app = FastAPI()

# Qwen model load kar
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class InputData(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(data: InputData):
    input_text = f"User: {data.prompt}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}

@app.get("/")
def home():
    return {"status": "Qwen API is running!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render free tier ke liye
    uvicorn.run("app:app", host="0.0.0.0", port=port)
