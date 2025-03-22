from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread
import time
import sys 
# ----------------------------------------------------------------------------:
model_name = "meta-llama/Llama-3.2-1B-Instruct"
#model_name = "meta-llama/Llama-3.2-3B-Instruct"
#model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# ----------------------------------------------------------------------------:

# ----------------------------------------------------------------------------:
# Set device
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
model = model.to(device)
print(f"DEBUG: Device set to: {device}")
# ----------------------------------------------------------------------------:


# ----------------------------------------------------------------------------:
# Example messages in OpenAI format
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]
chat_text = tokenizer.apply_chat_template(
    messages, 
    tokenize=True,
    add_generation_prompt=True
)
print(f"DEBUG: chat-text (post tokenizer.apply_chat_template):\n {chat_text}")
# ----------------------------------------------------------------------------:


sys.exit(1)
# https://huggingface.co/docs/transformers/main/en/chat_templating

# Tokenize the formatted chat
inputs = tokenizer(chat_text, return_tensors="pt").to(device)
input_tokens = inputs.input_ids.shape[1]

# Create streamer for token-by-token generation
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Measure prefill time
prefill_start = time.time()

# Start generation in a separate thread
generation_kwargs = {
    "input_ids": inputs.input_ids,
    "attention_mask": inputs.attention_mask,
    "max_new_tokens": 100,
    "temperature": 0.7,
    "streamer": streamer,
    "eos_token_id": None,
    "do_sample": True
}

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Get first token to measure prefill time
first_token = next(streamer)
prefill_end = time.time()
prefill_time = prefill_end - prefill_start

# Track generation time
generation_start = time.time()
generated_text = first_token
tokens_generated = 1

# Continue streaming tokens
for token in streamer:
    generated_text += token
    tokens_generated += 1

generation_end = time.time()
generation_time = generation_end - generation_start

# Print metrics
print(f"Input: {chat_text}")
print(f"Generated: {generated_text}")
print(f"Input tokens: {input_tokens}")
print(f"Output tokens: {tokens_generated}")
print(f"Prefill time: {prefill_time:.4f}s")
print(f"Generation time: {generation_time:.4f}s")
print(f"Generation tokens/second: {tokens_generated/generation_time:.2f}")
