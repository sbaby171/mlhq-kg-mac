from transformers import pipeline, GenerationConfig
import sys 

# QUES: How to get the Generation Config from an HF-CLIENT? 
# - Their results are always better. Check thier configs.

gen_config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True, 
)

model = "meta-llama/Llama-3.1-8B-Instruct"
task = "text-generation"
prompt = "Tell me a *mom* joke?"

pipe_org = pipeline(task=task, model=model)
for k,v in pipe_org.__dict__.items(): 
    if k == "tokenizer": 
        print(f"{k}  -->  <skipping>")
        continue 
    print(f"{k}  -->  {v}")
print(pipe_org(prompt))
print(f"\n\n{'-'*60}")


pipe_ctm = pipeline(task=task, model=model, generation_config=gen_config)
for k,v in pipe_ctm.__dict__.items(): 
    if k == "tokenizer": 
        print(f"{k}  -->  <skipping>")
        continue 
    print(f"{k}  -->  {v}")
print(pipe_ctm(prompt))
print(f"\n\n{'-'*60}")

messages=[
    {"role": "user", "content": f"{prompt}"}
  ]
print(pipe_ctm(messages))



sys.exit(0)





#pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", temperature=0)
# ValueError: `temperature` (=0) has to be a strictly positive float, otherwise your next token scores will be invalid.
pipe = pipeline(task="text-generation", 
    model="meta-llama/Llama-3.1-8B-Instruct",) 
print(pipe("Can you tell me a 'Mom' joke?", do_sample=False))
# UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
# UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
#pipe = pipeline(task="text-generation", 
#    model="meta-llama/Llama-3.1-8B-Instruct", 
#    do_sample=False, 
#    temperature = 0.0, 
#)
# UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
# UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
#print(pipe("Can you tell me a 'Mom' joke?"))


# ============================================================================:
#from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer separately
#model_name = "meta-llama/Llama-3.1-8B-Instruct"
#model = AutoModelForCausalLM.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#
## Create pipeline with explicit parameters
#pipe = pipeline(
#    task="text-generation",
#    model=model,
#    tokenizer=tokenizer,
#    do_sample=False,  # Use greedy decoding
#    # Don't specify temperature or top_p here
#)

#print(pipe("Can you tell me a 'Mom' joke?"))

# ============================================================================:
#from transformers import AutoModelForCausalLM, AutoTokenizer

## Load the model and tokenizer
#model_name = "meta-llama/Llama-3.1-8B-Instruct"
#model = AutoModelForCausalLM.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare input
#prompt = "Can you tell me a 'Mom' joke?"
#inputs = tokenizer(prompt, return_tensors="pt")

# Generate text deterministically using greedy search
#outputs = model.generate(
#    inputs.input_ids,
#    max_length=100,  # Set appropriate max length
#    do_sample=False,  # Greedy decoding (deterministic)
    # No need to specify temperature since do_sample=False
#)

# Decode the output
#generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print(generated_text)
