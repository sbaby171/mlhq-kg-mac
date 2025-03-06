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

    #def chat(self , model, messages, stream=False): 
    def chat(self, *args, **kwargs): 
        model = kwargs.get('model', self._model)
        messages = kwargs.get('messages', None)
        if messages == None: 
            raise RuntimeError("Must Provide `messages`.")
        stream = kwargs.get("stream", False)
        max_tokens = kwargs.get('max_tokens', 128)
      
        print(f"DEBUG [chat]: max-tokens={max_tokens}")
      
        if isinstance(self._client, ollama.Client): 
            return self._client.chat(model=model, messages=messages, stream=stream)
        elif isinstance(self._client, InferenceClient): 
            response = self._client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens, 
                stream=stream)
            return response
      
# --------------------------------------------------------------------|-------:
