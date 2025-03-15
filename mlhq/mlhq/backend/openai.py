import openai 
import ollama 
from huggingface_hub import InferenceClient
from transformers import pipeline #, AutoTokenizer, AutoConfig, AutoModelForCausalLM  
import os
from mlhq.config import load_model_registry
#from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Union, overlo
# --------------------------------------------------------------------|-------:
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BACKEND = "local" # pipelines
HF_LOCAL_BACKEND = "hf-local"
OLLAMA_BACKEND = "ollama"
HF_CLIENT_BACKEND = "hf-client"
#VLLM_BACKEND = "vllm"
#TRTLLM_BACKEND = "trtllm"
BACKENDS = [HF_LOCAL_BACKEND, OLLAMA_BACKEND, HF_CLIENT_BACKEND]
# --------------------------------------------------------------------|-------:

HF_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct", 
    "meta-llama/Llama-3.1-8B-Instruct", 
    "meta-llama/Meta-Llama-3-8B-Instruct", 
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
    #def __call__(self, *args, generate_kwargs={}):
        #print(f"DEBUG: [LazyPipeline]: args = {args}")
        #print(f"DEBUG: [LazyPipeline]: kwargs = {kwargs}")
        return self.pipeline(*args, **kwargs)
        #return self.pipeline(*args, generate_kwargs=generate_kwargs)
# --------------------------------------------------------------------|-------:
def _sync_model_and_backend(model, backend): 
    """ Check and aligns the model and backed. This function is used 
    within the Client creation.  
    """
    model_registry = load_model_registry()
    print(f"DEBUG: [sync-model-backend]: Model Registry Loaded = {model_registry}")

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- |-- --- :
    # If no model or backend, send the defaults. 
    # If model and backend, simply return.
    if (not model) and (not backend):
        return {'model':DEFAULT_MODEL,'backend':DEFAULT_BACKEND}
    if model and backend: 
        return {'model':model,'backend':backend}
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- |-- --- :

    if model in HF_MODELS: 
        try:
            hub_local = os.environ['HF_HOME']
        except KeyError:
            raise RuntimeError(f"Env variable 'HF_HOME' is not set.")
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



# ====================================================================|========:         
# NOTE - HuggingFace pipelines use GenerationConfig
class ClientParams: 
    def __init__(self, backend=None): 
        self._backend = backend 
        self._resolved = False 
        self.map = {} 
        self.stream = False 
        self.max_new_tokens = 128

    @property
    def stream(self): 
        return self._stream
    
    @stream.setter
    def stream(self, boolean):
        self._stream = boolean

    @property
    def max_new_tokens(self): 
        return self._max_new_tokens
    
    @max_new_tokens.setter
    def max_new_tokens(self, boolean):
        self._max_new_tokens = boolean

    def set_max_new_tokens(self, max_new_tokens): 
        self.max_new_tokens = max_new_tokens 


    #def __getattr__(self, name: str):
    #    return self.__dict__[f"{name}"]

    #def __setattr__(self, name, value):
    #    self.__dict__[f"{name}"] = value


    def is_resolved(self,): 
        return self._resolved
  
    def resolve(self, backend): 
        """The idea odf the resolve function is resolved the internal map 
        and for the user to access the values via the map.  
        """
        if backend == HF_CLIENT_BACKEND: # Backends.HF_CLIENT:  
            self.map["stream"] = self.stream
            self.map["max_new_tokens"] = self.max_new_tokens
        elif backend == HF_LOCAL_BACKEND: # Backends.HF_CLIENT:  
            self.map["stream"] = self.stream
            self.map["max_new_tokens"] = self.max_new_tokens
        else: 
            raise RuntimeError(f"Invalid backend '{backend}'")
    # ^^^ wait this function is not making much sense. If the user is going 
    # to use cparams.map["<name>"]. Instead, we can just pull directly, 
    # stream = cparams.stream 
        
            
  
            

 


  
# ====================================================================|========:         
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
        elif self._backend == HF_LOCAL_BACKEND: 
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
            print(f"DEBUG: Messages = {messages}")
            response = self._client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens, 
                stream=stream)
            return response
    # max_tokens for OpenAI compatiable 
    # max_new_tokens for text -generation
    # TODO: what is the difference between ChatCompletions and Text-Generation 

    # A method for completing conversations using a specified language model.
    
    # 
    #self.llm(self._build_agent_prompt()[1].content,
    #TypeError: 'Client' object is not callable
    #def __call__(self, messages, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=13, max_length=2048): 
    def __call__(self, messages, **kwargs): 
        # To make is callable implies high functionality. Very similiar to 
        # a HF Pipeline
        if isinstance(self._client, LazyPipeline): 
            return self._client(messages, **kwargs)

        return self.chat(messages = messages)

    def text_generation(self, prompt, cparams=None): 
        print(f"DEBUG: [Client.text_generation] Starting ... ")

        if not cparams.is_resolved(): 
            cparams.resolve(backend = self._backend) 

        if isinstance(self._client, InferenceClient): 
            if not cparams: 
                return self._client.text_generation(prompt)
                
            return self._client.text_generation(prompt, 
                max_new_tokens = cparams.map["max_new_tokens"], 
                stream = cparams.map["stream"], 
            )
        elif isinstance(self._client, LazyPipeline): 
            if not cparams: 
                return self._client(prompt)
            return self._client(prompt, 
                max_new_tokens = cparams.max_new_tokens, 
                #generate_kwargs = {
                #    #"max_new_tokens" : cparams.map["max_new_tokens"], 
                #    "max_new_tokens" : cparams.max_new_tokens, 
                #    #"max_new_tokens" : cparams.max_new_tokens, # This is why resolve is done
                #}
            )
       
        raise RuntimeError("Something bad")
      
# --------------------------------------------------------------------|-------:
