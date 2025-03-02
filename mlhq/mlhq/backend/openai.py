import openai 
import ollama 

# --------------------------------------------------------------------|-------:
class Client: 
    def __init__(self, *args, **kwargs): 
        self._client = None 
        self._backend = kwargs.get('backend')
        if not self._backend: 
            raise RuntimeError("Must provide 'backend' at constructor.")
        if self._backend == "ollama": 
            self._client = ollama.Client()
        elif self._backend == "vllm": 
            raise ValueError(f"Not supporting {self._backend} backend - atm.")
        elif self._backend in ["huggingface", "hf"]: 
            raise ValueError(f"Not supporting {self._backend} backend - atm.")
        else: 
            raise ValueError(f"Unrecognized {self._backend} backend")


    def chat(self , model, messages): 
        if self._backend == "ollama": 
            return self._client.chat(model, messages)
# --------------------------------------------------------------------|-------:
