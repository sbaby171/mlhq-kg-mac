import os, sys, re, argparse, json
from collections import OrderedDict
from mlhq.backend.openai import Client 
from mlhq.backend.openai import MODELS, HF_MODELS, OLLAMA_MODELS
# --------------------------------------------------------------------|-------: 
DEFAULT_QUES_JSON  = "/Users/msbabo/code/Graph-CoT/data/processed_data/biomedical/data.jsonl"
DEFAULT_GRAPH_JSON = "/Users/msbabo//code/Graph-CoT/data/processed_data/biomedical/graph.json"
DEFAULT_DB_URI     = 'mongodb://localhost:27017/'
DEFAULT_DB_NAME    = "graph-cot"
DEFAULT_DB_COLLECTION = "biomedical"
# --------------------------------------------------------------------|-------: 
def __handle_cli_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str, default="llama3.2:1b",choices = MODELS) 
    parser.add_argument('--questions-jsonl', type=str, default=DEFAULT_QUES_JSON)
    args = parser.parse_args()

    if not os.path.exists(args.questions_jsonl): 
        raise RuntimeError(f"Questions path not valid: {args.questions_json}")

    return args 
# --------------------------------------------------------------------|-------: 
def load_jsonl(jsonl): 
    data = [] 
    with open(jsonl, 'r') as file:
        for line in file:
            if line.strip():  
                data.append(json.loads(line))
    return data 
# --------------------------------------------------------------------|-------: 
def main(args): 
    results = OrderedDict() 

    ollama_model = True if (args.model in OLLAMA_MODELS) else False 

    client = Client(model=args.model)
    sys_prompt = "Can you answer the following question in a concise manner: "
    for qdata in args.questions: 
        qid = qdata['qid']
        ques = qdata['question']
        ans = qdata['answer']
        question = f"{sys_prompt}\n{ques}"
        response = client.chat(
             model = args.model, 
             messages = [{'role':'user', 'content': f"{question}"}],
             max_tokens = 512, 
             stream=False, 
        )
        print("question   : ", ques)
        print("prediction : ", response)
        results[qid] = OrderedDict()
        results[qid]['question'] = ques
        results[qid]['answer'] = ans
        if ollama_model: 
            results[qid]['prediction'] = response.message.content
        else: 
            results[qid]['prediction'] = response.choices[0].message
            

    with open("tmp.json", "w") as file:
        json.dump(results, file, indent=4)
# --------------------------------------------------------------------|-------:
if __name__ == "__main__": 
    args = __handle_cli_args()
    args.questions = load_jsonl(args.questions_jsonl) # qid, question, answer
    main(args)
