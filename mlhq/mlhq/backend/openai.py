import openai 
import ollama 
from huggingface_hub import InferenceClient
from transformers import pipeline #, AutoTokenizer, AutoConfig, AutoModelForCausalLM  
import os
from mlhq.config import load_model_registry
#from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Union, overlo
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread
import time
import sys
import os 
import numpy as np 
#import evaluate
from collections import OrderedDict
#from IPython.display import clear_output
from mlhq.utils import load_jsonl, write_json
from mlhq.utils import proceed 
from mlhq.utils import mean_reverse_diff
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
        


#model_name = "Qwen/Qwen2.5-7B-Instruct"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name)
def llm_generate(model, tokenizer, inputs, 
                 max_new_tokens = 128, 
                 temperature = None, 
                 top_p = None, 
                 top_k = None, 
                 repetition_penalty = None,
                 ignore_eos=False, 
                 do_sample = True,
                 stop_strings=None, 
        ):
        rd = {} # Return-Dictionary 
        streamer = TextIteratorStreamer(tokenizer, 
                                skip_prompt=True, 
                                skip_special_tokens=True)

        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "streamer": streamer,
        }

        if hasattr(model.generation_config, "max_new_tokens") and (not max_new_tokens): 
            max_new_tokens = model.generation_config.max_new_tokens
        if hasattr(model.generation_config, "temperature") and (not temperature): 
            temperature = model.generation_config.temperature
        if hasattr(model.generation_config, "top_p") and (not top_p): 
            top_p = model.generation_config.top_p
        if hasattr(model.generation_config, "top_k") and (not top_k): 
            top_k = model.generation_config.top_k
        if hasattr(model.generation_config, "repetition_penalty") and (not repetition_penalty): 
            repetition_penalty = model.generation_config.repetition_penalty
   
        gen_kwargs['tokenizer'] = tokenizer
        gen_kwargs['stop_strings'] = stop_strings
        gen_kwargs['do_sample'] = do_sample
        gen_kwargs["temperature"] = temperature  
        gen_kwargs["max_new_tokens"] = max_new_tokens  
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
        gen_kwargs["repetition_penalty"] = repetition_penalty 
        gen_kwargs['do_sample'] = model.generation_config.do_sample 
        gen_kwargs['bos_token_id'] = model.generation_config.bos_token_id
        gen_kwargs['eos_token_id'] = model.generation_config.eos_token_id
    
        if ignore_eos: 
            gen_kwargs["eos_token_id"] = None

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        prefill_start = time.time()
        first_token = next(streamer)
        prefill_end = time.time()
        prefill_time = prefill_end - prefill_start
        
        gen_text = first_token 
        num_gen_tokens = 1

        tbot_times = []
        gen_start = time.time()
    
        for token in streamer:
            tbot_times.append(time.time())
            gen_text += token
            num_gen_tokens += 1
        gen_end = time.time()
        gen_time =  gen_end - gen_start

        save_gen_kwargs = {}
        for k, v in gen_kwargs.items(): 
            if k in ["input_ids", "attention_mask", "streamer"]: continue 
            save_gen_kwargs[k] = v
            
        rd["prefill-time-s"]       = prefill_time
        rd["ttft-s"]               = prefill_time 
        rd["average-tbot-s"]       = mean_reverse_diff(tbot_times)
        rd["generation-time-s"]    = gen_time 
        rd["generated-text"]       = gen_text
        rd["num-generated-tokens"] = num_gen_tokens
        rd["generation-tps"]       = num_gen_tokens / gen_time
        rd["total-tps"]            = num_gen_tokens / (prefill_time + gen_time) 
        rd["generation-args"]      = save_gen_kwargs
        rd["total-latency-s"]      = prefill_time + gen_time
        rd["num-input-tokens"]     = inputs.input_ids.shape[1]
        return rd             
  
            
# ============================================================================:
class HFLocalClient:  
    def __init__(self, model_name): 
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if torch.cuda.is_available(): 
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
            self.device = "mps"
        else: 
            self.device = "cpu"
        self.model = self.model.to(self.device)
        print(f"DEBUG: Device set to: {self.device}")

    def generate(self, messages, **kwargs): 
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # RUN CLASSIFIER-LLM
        rd = llm_generate(self.model, self.tokenizer, inputs, **kwargs)
        # EXTRACT CLASSIFICATION
        return rd 

# ============================================================================:


  
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
    def __init__(self, model, backend, token=None): 
        self.model = model 
        self.backend = backend 

        if self.backend == Backends.OLLAMA: 
            self.client = ollama.Client()

        elif self.backend == Backends.HF_CLIENT: 
            self.client = InferenceClient(token=token)
            #self.client = InferenceClient(token=os.environ['HF_TOKEN'])

        elif self.backend == Backends.HF_LOCAL: 
            #self.client = LazyPipeline(model=self.model, task="text-generation")
            self.client = HFLocalClient(model)
        else:
            raise ValueError(f"Unrecognized {self._backend} backend")
     
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        # ^^^ TODO this will fail with Ollama 
    # ----------------------------------------------------------------|--------: 
    def text_generation(self, prompt, **kwargs): 
        if isinstance(prompt, list): # assuming its a list-dict openai-compatiable
             chat_text = self._tokenizer.apply_chat_template(                                      
                      prompt, 
                      tokenize=True,                                                              
                      add_generation_prompt=True)                                  
             prompt = self._tokenizer.decode(chat_text)
        return  self.client.text_generation(prompt, **kwargs)  
                                                                                
    # ----------------------------------------------------------------|--------: 
    def chat(self, *args, **kwargs): 
        model = kwargs.get('model', self.model)
        messages = kwargs.get('messages', None)
        if messages == None: 
            raise RuntimeError("Must Provide `messages`.")
        stream = kwargs.get("stream", False)
        max_tokens = kwargs.get('max_tokens', 128)
      
        print(f"DEBUG [chat]: max-tokens={max_tokens}")
      
        if isinstance(self.client, ollama.Client): 
            return self.client.chat(model=model, messages=messages, stream=stream)

        elif isinstance(self.client, LazyPipeline): 
            return self.client(messages)

        elif isinstance(self.client, InferenceClient): 
            print(f"DEBUG: Messages = {messages}")
            response = self.client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens, 
                stream=stream)
            return response
    # max_tokens for OpenAI compatiable 
    # max_new_tokens for text -generation
    # TODO: what is the difference between ChatCompletions and Text-Generation 


    # ------------------------------------------------------------------------:
    # TODO: An issue with this below is the fact that a HFModel.generate()
    #       does not require 
    def generate(self, messages, **kwargs): 
        if isinstance(self.client, HFLocalClient): 
            assert type(messages) == list, f"Only accepting Messages (list-of-dicts). Received type = {type(messages)}"
            # TODO: mlhq.utils.verify_messages_struct(messages)
            return self.client.generate(messages, **kwargs)
    
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
   
    #def text_generation(self, prompt, cparams=None): 
    #    if not cparams.is_resolved(): 
    #        cparams.resolve(backend = self.backend) 
    #
    #    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- |-- --- :
    #    # NOTE - I think HF-Client will automatically set the model based 
    #    #        on thae task. that is why I dont get an error at runtime
    #    #        when I don't pass the model in at runtime. 
    #    if isinstance(self.client, InferenceClient): 
    #        if not cparams: 
    #            return self.client.text_generation(prompt)
    #            
    #        return self.client.text_generation(prompt, 
    #            max_new_tokens = cparams.map["max_new_tokens"], 
    #            stream = cparams.map["stream"], 
    #        )
    #    elif isinstance(self.client, LazyPipeline): 
    #        if not cparams: 
    #            return self.client(prompt)
    #        return self.client(prompt, 
    #            #model = self.model, 
    #            #max_new_tokens = cparams.max_new_tokens, 
    #            #temperature = cparams.temperature,
    #            #do_sample = cparams.do_sample,
    #            **cparams.get_kwargs(), 
    #        )
    #    raise RuntimeError("Client class `{type(self.client)}` is unsupported.")
# --------------------------------------------------------------------|-------:
