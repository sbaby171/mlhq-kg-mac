from mlhq.backend.openai import Client 
from mlhq.backend.openai import MODELS
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
FIG_FONT = "chunky"#"larry3d"#"slant"
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
# ^^^ TODO - ollama models could be extracted from ~/.ollama
# --------------------------------------------------------------------|-------:
def __handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,choices = MODELS)
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
         if client.get_backend() == "ollama": 
             for chunk in response:
                 print(chunk['message']['content'], end='', flush=True)
         elif client.get_backend() == "huggingface": 
             for chunk in response: 
                 print(chunk.choices[0].delta.content, end='', flush=True)
         else: 
             raise RuntimeError("Unsupported backend: {client.get_backend()}")
# --------------------------------------------------------------------|-------:
if __name__ == "__main__": 
    args = __handle_cli_args()  
    main(args) 
