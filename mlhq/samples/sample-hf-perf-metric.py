import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer
from threading import Thread

# Load model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check for MPS (Apple Silicon)
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    model = model.to(device)
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Example prompt
system_prompt = "You are a helpful assistant."
user_question = "What is the capital of France?"
prompt = f"<s>[INST] {system_prompt} [/INST]</s>[INST] {user_question} [/INST]"

# Tokenize input
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
input_token_count = input_ids.shape[1]
print(f"Input tokens: {input_token_count}")

# Create streamer for token-by-token output
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Measure prefill time (time to process input and get first token)
prefill_start = time.time()

# Start generation in a separate thread
generation_kwargs = {
    "input_ids": input_ids,
    "max_new_tokens": 100,
    "temperature": 0.7,
    "streamer": streamer,
    "use_cache": True
}

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Get first token time (end of prefill)
first_token = next(streamer)
prefill_end = time.time()
prefill_time = prefill_end - prefill_start
print(f"Prefill time: {prefill_time:.4f}s")

# Track generation time and tokens
generation_start = time.time()
tokens_generated = 1  # Already got first token
generated_text = first_token

# Continue getting tokens and measuring
token_times = []
for token in streamer:
    #token_time = time.time()
    generated_text += token
    tokens_generated += 1
    token_time = time.time()
    token_times.append(token_time)

generation_end = time.time()
generation_time = generation_end - generation_start
total_time = prefill_time + generation_time

# Calculate metrics
generation_tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

print(f"Generated text: {generated_text}")
print(f"Generation tokens: {tokens_generated}")
print(f"Generation time: {generation_time:.4f}s")
print(f"Generation tokens per second: {generation_tokens_per_second:.2f}")
print(f"Total time (prefill + generation): {total_time:.4f}s")

# If you want to analyze token-by-token timing
if len(token_times) > 1:
    token_intervals = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
    avg_token_interval = sum(token_intervals) / len(token_intervals)
    print(f"Average time between tokens: {avg_token_interval:.6f}s")
    print(f"Steady-state tokens per second: {1.0/avg_token_interval:.2f}")
