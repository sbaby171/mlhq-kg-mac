import openai 
import ollama 
from huggingface_hub import InferenceClient
from transformers import pipeline #, AutoTokenizer, AutoConfig, AutoModelForCausalLM  
import os
# --------------------------------------------------------------------|-------:
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BACKEND = "local" # pipelines
LOCAL_BACKEND = "local"
VLLM_BACKEND = "vllm"
TRTLLM_BACKEND = "trtllm"
OLLAMA_BACKEND = "ollama"
HF_CLIENT_BACKEND = "hf-client"
BACKENDS = [LOCAL_BACKEND, OLLAMA_BACKEND, HF_CLIENT_BACKEND]
# --------------------------------------------------------------------|-------:

HF_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct", 
    "meta-llama/Llama-3.3-70B-Instruct", 
]
OLLAMA_MODELS = [
    "llama3.2",
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.2:latest",
    "llama3.3", 
    "llama3.3:latest",
    "deepseek-r1:8b", 
    "deepseek-r1:70b", 
]
MODELS = [*HF_MODELS,*OLLAMA_MODELS]
# --------------------------------------------------------------------|-------:
class LazyPipeline:
    def __init__(self, task, model=None, **kwargs):
        self.task = task
        self.model = model
        self.kwargs = kwargs
        self._pipeline = None
    
    @property
    def pipeline(self):
        if self._pipeline is None:
            print("Loading pipeline for the first time...")
            self._pipeline = pipeline(self.task, model=self.model, **self.kwargs)
        return self._pipeline
    
    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)
# --------------------------------------------------------------------|-------:
def _sync_model_and_backend(model, backend): 
    """ Check and aligns the model and backed. This function is used 
    within the Client creation.  
    """
    if (not model) and (not backend):
        return {'model':DEFAULT_MODEL,'backend':DEFAULT_BACKEND}

    if model in HF_MODELS: 
        try:
            hub_local = os.environ['HF_HOME']
        except KeyError:
            print(f"Env variable 'HF_HOME' is not set.")
        tmpname = "models--" + model.replace("/",'--')
        if tmpname in os.listdir(hub_local + "/hub"): 
            return {'model':model, "backend":LOCAL_BACKEND}
        else: # If not local --> HF_CLIENT
            print(f"INFO [sync-model]: Model '{model}'not found locally - setting backend to HF InferenceClient.")
            return {'model':model, "backend":HF_CLIENT_BACKEND}
    elif model in OLLAMA_MODELS: 
        print(f"DEBUG: [sync-model-backend]: Model {model}")
        return {'model':model, "backend": OLLAMA_BACKEND} 
    else: 
        raise ValueError(f"Model is not supported: {model}")
  
         
# TODO: I want a unit test to check the various options of the 
# constructor but in order to do so, I need to skip model Loading. 
# There is not need to test model loading. Therefore it needs to be 
# lazy loaded
# --------------------------------------------------------------------|-------:
class Client: 
    #def __init__(self, *args, **kwargs):  # Issues when: Client(model)
    #def __init__(self, **kwargs): 
    def __init__(self, *args, **kwargs): 
        self._client = None 
        if not args: 
            self._model   = kwargs.get('model', None)
        elif len(args) == 1: 
            self._model = args[-1] 
        self._backend = kwargs.get('backend', None)

        x = _sync_model_and_backend(self._model, self._backend)
        self._model = x['model']
        self._backend = x['backend']

        print(f"DEBUG: [client] model = {self._model}")
        print(f"DEBUG: [client] backend = {self._backend}")
        
        #if self._model in HF_MODELS: 
        #    self._backend = "huggingface" 
        #elif self._model in OLLAMA_MODELS: 
        #    self._backend = "ollama"

        if self._backend == OLLAMA_BACKEND: 
            self._client = ollama.Client()
        elif self._backend == HF_CLIENT_BACKEND: 
            self._client = InferenceClient(token=os.environ['HF_TOKEN'])
        elif self._backend == LOCAL_BACKEND: 
            self._client = LazyPipeline(model=self._model, task="text-generation")
        #elif self._backend == VLLM_BACKEND: 
        #    self._client == ...
        #elif self._backend == TRTLLM_BACKEND: 
        #    self._client == ...
        else:
            raise ValueError(f"Unrecognized {self._backend} backend")
   
        #if not self._backend: 
        #    raise RuntimeError("Must provide 'backend' at constructor.")
        #if self._backend == "ollama": 
        #    self._client = ollama.Client()
        #elif self._backend == "vllm": 
        #    raise ValueError(f"Not supporting {self._backend} backend - atm.")
        #elif self._backend in ["huggingface", "hf"]: 
        #    self._client = InferenceClient(token=os.environ['HF_TOKEN'])
        #    self._backend = "huggingface"
        #else: 

    
    def get_backend(self): 
        return self._backend
    @property 
    def backend(self): 
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

        elif isinstance(self._client, LazyPipeline): 
            return self._client(messages)

        elif isinstance(self._client, InferenceClient): 
            response = self._client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens, 
                stream=stream)
            return response
      
# --------------------------------------------------------------------|-------:
