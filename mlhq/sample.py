from mlhq.backend.openai import Client 
from mlhq.backend.openai import (
    MODELS, 
    OLLAMA_BACKEND, 
    LOCAL_BACKEND, 
    HF_CLIENT_BACKEND, 
    BACKENDS 
) 
# ^^^ improve to: from mlhq import Client
import argparse 
import sys 
import logging
from pyfiglet import Figlet
import torch
from transformers import pipeline, AutoTokenizer, AutoConfig 
# --------------------------------------------------------------------|-------:
FIG = Figlet(font="chunky") # larry3d, slant
print(FIG.renderText("MLHQ - Sample"))
# --------------------------------------------------------------------|-------:
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BACKEND = "local"
DEFAULT_PROMPT = "Tell me a dad joke."
DEFAULT_LOG_LEVEL = "info"
# ^^^ TODO - ollama models could be extracted from ~/.ollama
#pipe = pipeline(                                                                                                                                       
#                "text-generation",                                                 
#                model="meta-llama/Llama-3.1-8B-Instruct", 
#                torch_dtype=torch.float16,                                         
#                device_map="auto"                                                  
#       ) 
#messages = [{"role": "user", "content": "Tell me a dad joke."}]
#print(pipe(messages, do_sample = True, max_new_tokens=256))
#sys.exit(1)
# --------------------------------------------------------------------|-------:
# sample.py --model meta-llama/Llama-3.1-8B-Instruct
def __handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,choices = MODELS)
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND, choices=BACKENDS)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--no-stream', dest='stream', action='store_false')
    parser.add_argument('--log', type=str, default=DEFAULT_LOG_LEVEL, 
                                   help="Log level", 
                                   choices=["debug","info","warning"])
    parser.set_defaults(stream=True)
    args = parser.parse_args()

    if args.log == "debug": 
        args.logging = logging.basicConfig(level=logging.DEBUG)
    elif args.log == "info": 
        args.logging = logging.getLogger("main")
        args.logging.setLevel(logging.INFO)
    elif args.log == "info": 
        args.logging = logging.basicConfig(level=logging.WARNING)
    else: 
        raise RuntimeError(f"unsupported log-level: {args.log}")
    print(type(args.logging))
    return args
# --------------------------------------------------------------------|-------:
def main(args): 
     print(type(args.logging))
     args.logging.info(f"Received model: {args.model}")
     args.logging.info(f"Received backend: {args.backend}")
     print(f"Received model: {args.model}")
     print(f"Received backend: {args.backend}")

     #client = Client() 
     #client = Client(args.model)
     #client = Client(model = args.model)
     client = Client(model = args.model, backend=args.backend)
     response = client.chat(
         model = args.model, 
         messages = [{'role':'user', 'content': f"{args.prompt}"}],
         stream=args.stream, 
     )
     prompt_tokens = 0
     gen_tokens = 0 

     if client.backend == LOCAL_BACKEND: # issue within stream on HF Pipeline
         print(response)
     elif not args.stream: 
         print(response)
     else:
             
         if client.get_backend() == OLLAMA_BACKEND: 
             for chunk in response:
                 print(chunk['message']['content'], end='', flush=True)
             prompt_tokens = chunk["prompt_eval_count"]
             gen_tokens = chunk["eval_count"]
         elif client.backend == HF_CLIENT_BACKEND: 
             for chunk in response: 
                 print(chunk.choices[0].delta.content, end='', flush=True)
             prompt_tokens = "NA"
             gen_tokens = "NA"
         else: 
             raise RuntimeError("Unsupported backend: {client.get_backend()}")
         print("\n\nSummary of Execution:")
         print("prompt-tokens   : ", prompt_tokens)
         print("generated-tokens: ", gen_tokens)
     
# --------------------------------------------------------------------|-------:
if __name__ == "__main__": 
    args = __handle_cli_args()  
    main(args) 
