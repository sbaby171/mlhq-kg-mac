from mlhq.backend.openai import Client 
# ^^^ improve to: from mlhq import Client
import argparse 
# --------------------------------------------------------------------|-------:
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_BACKEND = "hf"
DEFAULT_PROMPT = "Hello, who are you?"
# --------------------------------------------------------------------|-------:
def __handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    args = parser.parse_args()
    return args

# --------------------------------------------------------------------|-------:
def main(args): 
     client = Client(backend=args.backend)
     response = client.chat(
         model = args.model, 
         messages = [{'role':'user', 'content': f"{args.prompt}"}]
     )
     print(response)
# --------------------------------------------------------------------|-------:
if __name__ == "__main__": 
    args = __handle_cli_args()  
    main(args) 
