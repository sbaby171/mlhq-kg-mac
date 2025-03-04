import openai 
import ollama 
from huggingface_hub import InferenceClient
#from transformers import AutoTokenizer, AutoModelForCausalLM
import os
HF_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct", 
]
OLLAMA_MODELS = [
    "llama3.2",
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.2:latest",
    "llama3.3", 
    "llama3.3:latest",
    "deepseek-r1:8b", 
]
MODELS = [*HF_MODELS,*OLLAMA_MODELS]
# --------------------------------------------------------------------|-------:
# Note - Its probably the case that model is not declared until the 
#        chat functino within the OpenAI API because users could swap
#        to different models on a given prompt. Therefore, we should 
#        follow this and change only for good reason.
class Client: 
    def __init__(self, *args, **kwargs): 
        self._client = None 
        self._backend = kwargs.get('backend')
        self._model = kwargs.get('model')

        if self._model in HF_MODELS: 
            self._backend = "huggingface" 
        elif self._model in OLLAMA_MODELS: 
            self._backend = "ollama"

        print(f"DEBUG: [client] model = {self._model}")
        print(f"DEBUG: [client] backend = {self._backend}")
        if not self._backend: 
            raise RuntimeError("Must provide 'backend' at constructor.")
        if self._backend == "ollama": 
            self._client = ollama.Client()
        elif self._backend == "vllm": 
            raise ValueError(f"Not supporting {self._backend} backend - atm.")
        elif self._backend in ["huggingface", "hf"]: 
            self._client = InferenceClient(token=os.environ['HF_TOKEN'])
            self._backend = "huggingface"
        else: 
            raise ValueError(f"Unrecognized {self._backend} backend")


    def get_backend(self): 
        return self._backend

    def chat(self , model, messages, stream=True): 
    #def chat(self , *args, **kwargs): 
        if isinstance(self._client, ollama.Client): 
            return self._client.chat(self._model, messages, stream=stream)
            #return self._client.chat(messages, stream=stream)
        elif isinstance(self._client, InferenceClient): 
            #response = self._client.chat_completion(messages, max_tokens=100)
            #response = self._client.chat_completion(messages, max_tokens=100, stream=stream)
            response = self._client.chat_completion(model=model, messages=messages, max_tokens=100, stream=stream)
            return response
            #return self._client.chat.completions.create(model=model, messages=messages, stream=stream)
      
# --------------------------------------------------------------------|-------:
