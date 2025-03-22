from transformers import pipeline, GenerationConfig, AutoTokenizer
import sys
import time  
import torch
from datetime import datetime
from mlhq.utils import load_jsonl, write_json, get_datetime_str, proceed
"""
Note, while working on LLMs Inference research and develope, you need 
safe guards to save your work as you go. For example, storing all results
in a dictionary to write-to-file at the end is bad news. You need to be 
saving all your results as you go to disk, just in case something goes 
wrong. 

This may be why console logs with a `2>&1 | tee`
"""
skip_proceed = False 
# Check if MPS is available (for Apple Silicon Macs)
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")


# ----------------------------------------------------------------------------:
model = "meta-llama/Llama-3.1-8B-Instruct"
model = "meta-llama/Llama-3.2-1B-Instruct"
task = "text-generation"
tokenizer = AutoTokenizer.from_pretrained(model)
#gen_config = GenerationConfig(                                                  
#    max_new_tokens=200,                                                         
#    temperature=0.6,
#    do_sample=True,                                                             
#) 
pipe = pipeline(task=task, model=model)
# ----------------------------------------------------------------------------:
qa_paths = {
    "biomedical" : "/Users/msbabo/code/Graph-CoT/data/processed_data/biomedical/data.jsonl", 
    "amazon" : "/Users/msbabo/code/Graph-CoT/data/processed_data/amazon/data.jsonl", 
    "dblp" : "/Users/msbabo/code/Graph-CoT/data/processed_data/dblp/data.jsonl", 
    "goodreads": "/Users/msbabo/code/Graph-CoT/data/processed_data/goodreads/data.jsonl", 
    "legal": "/Users/msbabo/code/Graph-CoT/data/processed_data/legal/data.jsonl", 
    "maple-biology": "/Users/msbabo/code/Graph-CoT/data/processed_data/maple/Biology/data.jsonl",
    "maple-medicine": "/Users/msbabo/code/Graph-CoT/data/processed_data/maple/Medicine/data.jsonl",
    "maple-physics": "/Users/msbabo/code/Graph-CoT/data/processed_data/maple/Physics/data.jsonl",
    "maple-materials-science": "/Users/msbabo/code/Graph-CoT/data/processed_data/maple/Materials_Science/data.jsonl",
    "maple-chemistry": "/Users/msbabo/code/Graph-CoT/data/processed_data/maple/Chemistry/data.jsonl",
}
domains = [ 
    "amazon", 
    "biomedical", 
    "dblp", 
    "goodreads", 
    "legal", 
    "maple-biology", 
    "maple-physics", 
    "maple-chemisty", 
    "maple-medcine", 
    "maple-materials-science", 
]
# ----------------------------------------------------------------------------:
system_prompt = """
You are an expert prompt classifier designed to categorize user prompts accurately into one of two distinct categories:

Internal Response: Prompts for which an LLM can confidently provide accurate, complete, and helpful responses solely using its internal knowledge, without needing external information, databases, APIs, or tooling.

External Response: Prompts for which an LLM requires external knowledge, updated information, APIs, databases, or external tooling to confidently deliver accurate, complete, and helpful responses.

Guidelines for Classification:

Classify prompts as "Internal Response" if the information needed to answer fully and confidently is within general world knowledge as of the training cutoff, and no specific, updated, or external information is required.

Classify prompts as "External Response" if answering correctly and confidently necessitates:

Real-time data, current events, or recent updates.

Information specific to locations, current weather, dates, times, recent publications, or market values.

External computation, databases, or web searches.

Provide only the classification label ("Internal Response" or "External Response") as your output.
"""
system_prompt_len = len(pipe.tokenizer(system_prompt)['input_ids'])

prefix = "Incoming Prompt:\n"
prefix_len = len(pipe.tokenizer(prefix)['input_ids'])
# ----------------------------------------------------------------------------:
results = {}
results['model'] = model 
results['task'] = task 
results['forward-params'] = dict(pipe.__dict__["_forward_params"])
results["generation-config"] = {} 
results["system-prompt"] = system_prompt
results["system-prompt-len"] = system_prompt_len
results["prefix"] = prefix
results["prefix-len"] = prefix_len
print(f"System prompt length = {system_prompt_len}")
print(f"Prefix length = {prefix_len}")
proceed(skip=skip_proceed)
# ----------------------------------------------------------------------------:


for k, v in pipe.generation_config.__dict__.items(): 
    if type(v) not in [str, None, bool, float, list, dict]: continue 
    results["generation-config"][k] = v 
results['model-config'] = dict(pipe.model.config.__dict__) 
dt = get_datetime_str() 
for domain in domains: 
    qa_data = load_jsonl(qa_paths[domain])
    results[domain] = [] 
    count = 10


    Model = pipe.model
    for i, qdata in enumerate(qa_data, start=1): 
        if i > count: break 
        qid = qdata["qid"]
        ques = qdata["question"]
        messages = [
        {
            "role": "system",
            "content": f"{system_prompt}"
        },
        {
            "role": "user", 
            "content": f"{prefix} {ques}"
        }
        ]
        #if hasattr(pipe, "tokenizer"):
        #    prompt_text = ""
        #    for msg in messages:
        #        prompt_text += f"{msg['role']}: {msg['content']}\n"
        #    num_input_tokens = len(pipe.tokenizer(prompt_text)['input_ids'])
        #else:
        #    prompt_text = ""
        #    for msg in messages:
        #        prompt_text += f"{msg['role']}: {msg['content']}\n"
        #    num_input_tokens = len(tokenizer(prompt_text)['input_ids'])

        prompt_text = ""
        for msg in messages:
            prompt_text += f"{msg['role']}: {msg['content']}\n"
        num_input_tokens = len(pipe.tokenizer(prompt_text)['input_ids'])
        # --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  :
        prefill_start = time.time()
        inputs = tokenizer(prompt_text, return_tensors="pt")
        with torch.no_grad():
            outputs = Model(input_ids=inputs['input_ids'])
        prefill_end = time.time()
        prefill_time = prefill_end - prefill_start
        # --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  :

        # --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  :
        gen_start = time.time()
        response = pipe(messages)
        gen_end = time.time()
        # --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  :

        #total_time = gen_end - prefill_start
        total_time = gen_end - gen_start
        generation_time = total_time - prefill_time

        classified = response[0]['generated_text'][-1]['content']
        num_output_tokens = len(tokenizer(classified)['input_ids'])

        #tokens_per_second = num_output_tokens / inference_time if inference_time > 0 else 0
        generation_tokens_per_second = num_output_tokens / generation_time if generation_time > 0 else 0

        # client.chat(messages=messages)
        # client.last_response(raw=True)
        # client.last_response(logits=True)
        # client.response(last=1, raw=1)
        # --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  :
        print(f"{i}.) [{domain}] Question   --> {ques}")
        print(f"{i}.) [{domain}] Classified --> {classified}")
        print(f"  - Input tokens: {num_input_tokens}, Output tokens: {num_output_tokens}")
        print(f"  - SP+PX tokens: {system_prompt_len + prefix_len}")
        print(f"  -  QUES tokens: {len(pipe.tokenizer(ques)['input_ids'])}")
        #print(f"    Inference time: {inference_time:.2f}s, Tokens/second: {tokens_per_second:.2f}")
        print(f"    Input tokens: {num_input_tokens}")
        print(f"    Output tokens: {num_output_tokens}")
        print(f"    Prefill time: {prefill_time:.4f}s")
        print(f"    Generation time: {generation_time:.4f}s")
        print(f"    Total time: {total_time:.4f}s")
        print(f"    Generation tokens/second: {generation_tokens_per_second:.2f}")
        if "Internal Response" in classified: 
            classified = "Internal Response"
        elif "External Response" in classified: 
            classified = "External Response"
        else: 
            raise RuntimeError("bad llm response")
        # SimpleAgent("sys + prefix + prompt").SimpleAgent("sys2 + prefix2")
        results[domain].append({'qid':qid, 'question':ques, "classified":classified})
        proceed(skip=skip_proceed)
     
revised_model_name = model.replace('/','--')
write_json(results, f"./baseline-{revised_model_name}-{dt}.json")
