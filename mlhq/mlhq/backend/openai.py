import openai 
import ollama 
from huggingface_hub import InferenceClient
#from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# --------------------------------------------------------------------|-------:
class Client: 
    def __init__(self, *args, **kwargs): 
        self._client = None 
        self._backend = kwargs.get('backend')
        self._model = kwargs.get('model')
        if not self._backend: 
            raise RuntimeError("Must provide 'backend' at constructor.")
        if self._backend == "ollama": 
            self._client = ollama.Client()
        elif self._backend == "vllm": 
            raise ValueError(f"Not supporting {self._backend} backend - atm.")
        elif self._backend in ["huggingface", "hf"]: 
            self._client = InferenceClient(model=self._model, token=os.environ['HF_TOKEN'])
            self._backend = "huggingface"
        else: 
            raise ValueError(f"Unrecognized {self._backend} backend")


    def chat(self , model, messages, stream=False): 
    #def chat(self , *args, **kwargs): 
        if isinstance(self._client, ollama.Client): 
            print(f"DEBUG: [chat] - Ollamabakend")
            return self._client.chat(model, messages, stream=stream)
        elif isinstance(self._client, InferenceClient): 
            #response = self._client.chat_completion(messages, max_tokens=100)
            response = self._client.chat_completion(messages, max_tokens=100, stream=stream)
            return response
            #return self._client.chat.completions.create(model=model, messages=messages, stream=stream)
      
# --------------------------------------------------------------------|-------:
