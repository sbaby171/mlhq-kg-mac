from mlhq.backend.openai import Client 
# ^^^ improve to: from mlhq import Client
import argparse 
import sys 
# --------------------------------------------------------------------|-------:
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BACKEND = "hf"
DEFAULT_PROMPT = "Hello, who are you?"

HF_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct", 
]
OLLAMA_MODELS = [
    "deepseek-r1:8b", 
]
# --------------------------------------------------------------------|-------:
def __handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--no-stream', dest='stream', action='store_false')
    parser.set_defaults(stream=True)
    args = parser.parse_args()

    if args.model in HF_MODELS: 
        args.backend = "huggingface" 
    elif args.model in OLLAMA_MODELS: 
        args.backend = "ollama"
    # ^^^ NOTE - this check could be pushed into mlhq.backend.openai.py
    return args
# --------------------------------------------------------------------|-------:
def main(args): 
     print(f"DEBUG: [main] Received model: {args.model}")
     print(f"DEBUG: [main] Received backend: {args.backend}")
     #sys.exit(1)
     client = Client(model = args.model, backend=args.backend)
     response = client.chat(
         model = args.model, 
         messages = [{'role':'user', 'content': f"{args.prompt}"}],
         stream=args.stream, 
     )
     if args.stream: 
         if args.backend == "ollama": 
             for chunk in response:
                 print(chunk['message']['content'], end='', flush=True)
         elif args.backend == "huggingface": 
             for chunk in response: 
                 #print(chunk.content, end='', flush=True)
                 print(chunk.choices[0].delta.content, end='', flush=True)
     else: 
         print(response)
# --------------------------------------------------------------------|-------:
if __name__ == "__main__": 
    args = __handle_cli_args()  
    main(args) 
