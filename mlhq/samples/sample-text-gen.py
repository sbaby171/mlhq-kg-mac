#from mlhq.backend.openai import Client
#from mlhq.backend import Client
from mlhq import Client, ClientParams, Backends
from pyfiglet import Figlet
import argparse
import sys 

# ============================================================================:
def __handle_cli_args(): 
     parser = argparse.ArgumentParser()
     #parser = parser.add_argument("--model", type=str, choices=Models.choices())
     parser.add_argument("--backend", type=str, 
         default = Backends.HF_CLIENT,
         choices=Backends.choices()
     )
     # ^^^ TODO: See both of these should be coming from the model-registry 
     args = parser.parse_args()
     return args

# ============================================================================:
if __name__ == "__main__": 
    args = __handle_cli_args() 

    FIG = Figlet(font="slant") # larry3d, slant, chunky
    print(FIG.renderText("MLHQ - TexGen")) 

    model = "meta-llama/LLama-3.1-8B-Instruct"
    backend = args.backend  # "hf-local"

    #client = Client(model=model, backend=backend)
    #client = Client(backend=backend)
    client = Client(model, backend)
    print(type(client))
    print("Client.model = ", client.model)
    print("Client.backend = ", client.backend)


    cparams = ClientParams() 
    cparams.max_new_tokens = 128 
    cparams.temperature = 0.0
    cparams.do_sample = False 
    prompt = "Can you tell me a *mom* joke?"
    response = client.text_generation(prompt, cparams)
    print(response)

    cparams.max_new_tokens = 200
    cparams.temperature = 1.0
    cparams.do_sample = True
    response = client.text_generation(prompt, cparams)
    print(response)
    sys.exit(1)

# Note - OpenAI API doesn't have a `text-generation`. They simply use
#        a Chat (create completions) for that. But HuggingFace Inference
#        client does have a text-generation. 

# ============================================================================:
# >>> client.text_generation("The huggingface_hub library is ", max_new_tokens=12)
# IN order to has a portal that support multiple backends (hf-local, 
# pipelines, ollama, hf-inference client, vllm, etc), it is probably
# best to push all function args into a class like GenerationParams,
# then the mlhq.Client handles mapping that for the particular backend.  
# 
# Below are a few examples of how a ClientParams class can be constructed
# and updated. For now, we will implement case 2. It is the simpliest 
# to implement IMO. 
# 
# I think a key question is, "When should the backend provided". Well,
# if you say that the backend must be provided at Client creation, then 
# its not unreasonable to not enfore that at ClientParam creation. 
# However, it might be best that this is resolved at the first inference, 
# when the params is passed in. However, this technically adds latency 
# on first inference (check + resolve) then every other inference (check). 
# It might be neligble. Note, when the cparams is passed into Client, 
# the client knows its backend so it can be passed to Cparams.  
# 
# assigning the parameters to the Client class itself, might feel like a 
# good idea, but its now static. And if you want to still support users 
# passing into values at calls, then you wil run into the multiple names
# from different backend ends again. 
# 
# Another execution method you can have is is to `resolve` at pass-in.
# For example, `client.chat(messages, cparams.resolve() )`, but in this 
# case the user needs to remeber and where does the backend signel come 
# from? So it really needs to be 
# `client.chat(messages, cparams.resolve(backend))` - thats way too much
# 
# For now, lets pass it in, and do a check at inference. 
# 
#from mlhq import ClientParams

# Case -1 
# cparams = ClientParams(max_new_tokens = 128)

# Case - 2
#cparams = ClientParams() 
#cparams.set_max_new_tokens(128)
#cparams.stream = True 
#cparams.max_new_tokens = 128 

# Case - 3 
# params = {"max_new_tokens":128}
# cparams = ClientParams(params)

# Case - 4
# cparams = ClientParams()
# params = {"max_new_tokens":128}
# cparams.set_params(params)

#response = client.text_generation("The huggingface_hub library is ", cparams)
#print(response)
# https://huggingface.co/docs/huggingface_hub/guides/inference#openai-compatibility 


