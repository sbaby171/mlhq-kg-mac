import openai 
import ollama 
from huggingface_hub import InferenceClient
from transformers import pipeline #, AutoTokenizer, AutoConfig, AutoModelForCausalLM  
import os
from mlhq.config import load_model_registry
#from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Union, overlo
# ====================================================================|=======:
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BACKEND = "local" # pipelines
HF_LOCAL_BACKEND = "hf-local"
OLLAMA_BACKEND = "ollama"
HF_CLIENT_BACKEND = "hf-client"
#VLLM_BACKEND = "vllm"
#TRTLLM_BACKEND = "trtllm"
BACKENDS = [HF_LOCAL_BACKEND, OLLAMA_BACKEND, HF_CLIENT_BACKEND]
# ====================================================================|=======:
class Backends:
    HF_LOCAL  = "hf-local"
    HF_CLIENT = "hf-client"
    OLLAMA    = "ollama"
   
    @staticmethod
    def choices(): 
        return [Backends.HF_LOCAL, Backends.HF_CLIENT, Backends.OLLAMA] 
# ====================================================================|=======:

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
            print(f"  - model = {self.model}")
            print(f"  - task = {self.task}")
            self._pipeline = pipeline(self.task, model=self.model, **self.kwargs)
        return self._pipeline
    
    def __call__(self, *args, **kwargs):
    #def __call__(self, *args, generate_kwargs={}):
        #print(f"DEBUG: [LazyPipeline]: args = {args}")
        #print(f"DEBUG: [LazyPipeline]: kwargs = {kwargs}")
        print("issing LazyPipeline.__call__")
        print(f"args = {args}")
        print(f"kwargs = {kwargs}")


        #if not self.model: 
        #    if "model" in kwargs: 
        #        self.model = kwargs["model"]
        #    else: 
        #        raise RuntimeError("Need model name")
        return self.pipeline(*args, **kwargs)
        #return self.pipeline(*args, generate_kwargs=generate_kwargs)
# --------------------------------------------------------------------|-------:
def _sync_model_and_backend(model, backend): 
    """ Check and aligns the model and backed. This function is used 
    within the Client creation.  
    """
    model_registry = load_model_registry()
    for model_name , details in model_registry.items(): 
        print(model_name, details )
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
        self.temperature = 1.0 
        self.do_sample = True 
   
    def get_kwargs(self,): 
        #if not self._backend: 
        #    return {"stream":self.stream, "max_new_tokens"}
        if self._backend == Backends.HF_LOCAL: 
            if self.temperature == 0.0: 
                self.do_sample = False
                return {"do_sample":self.do_sample,
                        "max_new_tokens": self.max_new_tokens,
                }

    @property
    def stream(self): 
        return self._stream
    
    @stream.setter
    def stream(self, boolean):
        self._stream = boolean

    # ----------------------------------------------------------------|-------:
    @property
    def max_new_tokens(self): 
        return self._max_new_tokens
    
    @max_new_tokens.setter
    def max_new_tokens(self, value):
        self._max_new_tokens = value

    #def set_max_new_tokens(self, max_new_tokens): 
    #    self.max_new_tokens = max_new_tokens 
    # ----------------------------------------------------------------|-------:
    @property
    def temperature(self): 
        return self._temperature
    
    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    #def set_temperature(self, value): 
    #    self.temperature = value
    # ----------------------------------------------------------------|-------:
    @property
    def do_sample(self): 
        return self._do_sample
    
    @do_sample.setter
    def do_sample(self, value):
        self._do_sample = value

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
        self._backend = backend
 
        #if backend == HF_CLIENT_BACKEND: # Backends.HF_CLIENT:  
        #    self.map["stream"] = self.stream
        #    self.map["max_new_tokens"] = self.max_new_tokens
        #    
        #elif backend == HF_LOCAL_BACKEND: # Backends.HF_CLIENT:  
        #    self.map["stream"] = self.stream
        #    self.map["max_new_tokens"] = self.max_new_tokens
        #else: 
        #    raise RuntimeError(f"Invalid backend '{backend}'")
    # ^^^ wait this function is not making much sense. If the user is going 
    # to use cparams.map["<name>"]. Instead, we can just pull directly, 
    # stream = cparams.stream 
        
            
  
            

 


  
# ====================================================================|========:         
# TODO: I want a unit test to check the various options of the 
# constructor but in order to do so, I need to skip model Loading. 
# There is not need to test model loading. Therefore it needs to be 
# lazy loaded
# 
# NOTE: there is going to be an issue with trying to consildate Clients
#       and HF pipelines. The pipelines typically need a model at Creation
#       time. 
# 
# For now we will force users to ppass
#=====================================================================|=======:
class Client: 
    #def __init__(self, model="", backend=Backends.HF_LOCAL): 
    def __init__(self, model, backend): 
        self.model = model 
        self.backend = backend 
        #self.client = None 

        if self.backend == Backends.OLLAMA: 
            self.client = ollama.Client()

        elif self.backend == Backends.HF_CLIENT: 
            self.client = InferenceClient(token=os.environ['HF_TOKEN'])

        elif self.backend == Backends.HF_LOCAL: 
            self.client = LazyPipeline(model=self.model, task="text-generation")

        else:
            raise ValueError(f"Unrecognized {self._backend} backend")

    # ----------------------------------------------------------------|--------: 
    @property 
    def backend(self): 
        return self._backend 

    @backend.setter
    def backend(self, backend):
        self._backend = backend
    # ----------------------------------------------------------------|--------: 

    # ----------------------------------------------------------------|--------: 
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
    # ----------------------------------------------------------------|-------:
    def __call__(self, messages, **kwargs): 

        if isinstance(self._client, LazyPipeline): 
            return self._client(messages, **kwargs)

        return self.chat(messages = messages)
    # ----------------------------------------------------------------|-------:

    # ----------------------------------------------------------------|-------:
    # NOTE - the issues I am running into is that Client does not need the 
    #       
    def text_generation(self, prompt, cparams=None): 

        if not cparams.is_resolved(): 
            cparams.resolve(backend = self.backend) 

        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- |-- --- :
        # NOTE - I think HF-Client will automatically set the model based 
        #        on thae task. that is why I dont get an error at runtime
        #        when I don't pass the model in at runtime. 
        if isinstance(self.client, InferenceClient): 
            if not cparams: 
                return self.client.text_generation(prompt)
                
            return self.client.text_generation(prompt, 
                max_new_tokens = cparams.map["max_new_tokens"], 
                stream = cparams.map["stream"], 
            )
        elif isinstance(self.client, LazyPipeline): 
            if not cparams: 
                return self.client(prompt)
            return self.client(prompt, 
                #model = self.model, 
                #max_new_tokens = cparams.max_new_tokens, 
                #temperature = cparams.temperature,
                #do_sample = cparams.do_sample,
                **cparams.get_kwargs(), 
            )
        raise RuntimeError("Client class `{type(self.client)}` is unsupported.")
# --------------------------------------------------------------------|-------:
