import gradio as gr
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

# Directory is the same as app.py
MODEL_DIR = "."  # or os.path.dirname(__file__)

try:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded successfully from local directory!")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    print(f"Attempting to load from a potentially higher level directory or check MODEL_DIR: {MODEL_DIR}")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("Loaded default GPT2 model and tokenizer as a fallback.")

generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

def translate_shakespeare(shakespeare_text):
    prompt = f"SOURCE: {shakespeare_text} TARGET:"
    output = generator(
        prompt,
        max_new_tokens=30,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )
    # The output is a list of dictionaries, so we index the first element
    generated_text = output[0]['generated_text'] 
    translation = generated_text.replace(prompt, '').replace(tokenizer.eos_token, '').strip()
    return translation

iface = gr.Interface(
    fn=translate_shakespeare,
    inputs=gr.Textbox(lines=2, placeholder="Enter Shakespearean text here..."),
    outputs=gr.Textbox(label="Modern English Translation"),
    title="Shakespearean to Modern English Translator",
    description="LLM that translates phrases from Shakespearean English to Modern English using GPT-2. || IAT360 Spring 2025",
)

if __name__ == "__main__":
    # Pass the theme argument to the launch() method
    iface.launch(theme=gr.themes.Monochrome())
