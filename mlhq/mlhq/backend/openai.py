import openai 
import ollama 
from huggingface_hub import InferenceClient
#from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# --------------------------------------------------------------------|-------:
class HFClient: 
    def __init__(self, *args, **kwargs): 
        self._backend = "huggingface"
        
    def chat(self, model, messages, stream=False): 
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model) 
        #return model(messages)
        # Tokenize the input messages and return the result in tensor form
        #inputs = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)
        #
        ## Forward the tokenized inputs to the model and get the response
        #outputs = model.generate(**inputs, max_length=100)  # Use model.generate for generation
        #
        # Decode the model's output tokens back to text
        #response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #return response
        # Extract the 'content' from each message in the list of dictionaries
        message_texts = [message['content'] for message in messages]
        
        # Tokenize the list of message texts
        inputs = tokenizer(message_texts, return_tensors="pt", padding=True, truncation=True)
        
        # Generate model response using tokenized inputs
        outputs = model.generate(**inputs, max_length=100)
        
        # Decode the model's output tokens back to text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response

        

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
            #raise ValueError(f"Not supporting {self._backend} backend - atm.")
            #self._client = HFClient()
            self._client = InferenceClient()
            self._backend = "huggingface"
        else: 
            raise ValueError(f"Unrecognized {self._backend} backend")


    def chat(self , model, messages, stream=False): 
        if self._backend == "ollama": 
            return self._client.chat(model, messages, stream=stream)
        elif self._backend == "huggingface": 
            return self._client.chat.completions.create(model=model, messages=messages, stream=stream)
      
# --------------------------------------------------------------------|-------:
