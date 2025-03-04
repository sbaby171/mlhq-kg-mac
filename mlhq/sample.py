from mlhq.backend.openai import Client 
# ^^^ improve to: from mlhq import Client
import argparse 
import sys 
import logging
from pyfiglet import Figlet
# --------------------------------------------------------------------|-------:
RED = '\033[31m'
BLUE = '\033[34m'
YELLOW = '\033[33m'
BOLD = '\033[1m'
RESET = '\033[0m'
FIG_FONT = "slant"
FIG = Figlet(font=FIG_FONT)
#banner = f"{BOLD}{BLUE}MLHQ - Sample"
banner = f"MLHQ - Sample"
#print(f"{BOLD}{RED}Error:{RESET} Something went wrong")
#print(f"{BOLD}{BLUE}Info:{RESET} Processing data...")
#print(f"{YELLOW}Warning:{RESET} Disk space is low")
print(FIG.renderText(banner))
# --------------------------------------------------------------------|-------:
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BACKEND = "hf"
DEFAULT_PROMPT = "Hello, who are you?"
DEFAULT_LOG_LEVEL = "info"
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
# ^^^ TODO - ollama models could be extracted from ~/.ollama
# --------------------------------------------------------------------|-------:
def __handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND)
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

    if args.model in HF_MODELS: 
        args.backend = "huggingface" 
    elif args.model in OLLAMA_MODELS: 
        args.backend = "ollama"
    # ^^^ NOTE - this check could be pushed into mlhq.backend.openai.py
    return args
# --------------------------------------------------------------------|-------:
def main(args): 
     print(type(args.logging))
     args.logging.info(f"Received model: {args.model}")
     args.logging.info(f"Received backend: {args.backend}")
     print(f"Received model: {args.model}")
     print(f"Received backend: {args.backend}")
     client = Client(model = args.model, backend=args.backend)
     response = client.chat(
         model = args.model, 
         messages = [{'role':'user', 'content': f"{args.prompt}"}],
         stream=args.stream, 
     )
     if not args.stream: 
         print(response)
     else:
         if args.backend == "ollama": 
             for chunk in response:
                 print(chunk['message']['content'], end='', flush=True)
         elif args.backend == "huggingface": 
             for chunk in response: 
                 print(chunk.choices[0].delta.content, end='', flush=True)
# --------------------------------------------------------------------|-------:
if __name__ == "__main__": 
    args = __handle_cli_args()  
    main(args) 
