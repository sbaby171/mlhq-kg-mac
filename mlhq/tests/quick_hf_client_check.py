from huggingface_hub import InferenceClient

client = InferenceClient()
messages = [{"role": "user", "content": "Tell me a dad joke?".strip()}]

#model =  "meta-llama/Llama-3.1-8B-Instruct"
model =  "meta-llama/Meta-Llama-3-8B-Instruct" # GOOD
model =  "meta-llama/Meta-Llama-3-70B-Instruct" # GOOD
#model =  "meta-llama/Meta-Llama-3.1-8B-Instruct" # BAD
response = client.chat_completion(
    model=model,
    messages=messages, 
    stream=False,
)

print(response)

    #tokenizer_config_path="/Users/msbabo/.cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/tokenizer.json"
