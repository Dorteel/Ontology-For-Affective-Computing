

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests, os

model_id = "Qwen/Qwen2-VL-2B-Instruct"
device = "cpu"
dtype = torch.bfloat16

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
    attn_implementation="eager",
).eval()

#img_path = "/home/kai/Repositories/Ontology-For-Affective-Computing/databases/emotions_database/train/disgusted/im257.png"
img_path = '/home/kai/Repositories/Ontology-For-Affective-Computing/databases/serenity.png'
img = Image.open(img_path).convert("RGB")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text":
             "Look at the image and describe the person's facial expression and likely emotion(s)."},
        ],
    }
]


# Build the text that contains the image token(s)
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Prepare inputs (pass the image too)
inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)

with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    # Trim the prompt tokens for clean decoding
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    print(processor.batch_decode(trimmed, skip_special_tokens=True)[0])
