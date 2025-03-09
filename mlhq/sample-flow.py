import os, sys, re, argparse, json, jsonlines
from collections import OrderedDict
from mlhq.backend.openai import Client 
from mlhq.backend.openai import MODELS, HF_MODELS, OLLAMA_MODELS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- |-- --- :
from GraphAgent import GraphAgent
from tools.retriever import NODE_TEXT_KEYS
from graph_prompts import graph_agent_prompt, graph_agent_prompt_zeroshot 
# --------------------------------------------------------------------|-------: 
DEFAULT_QUES_JSON  = "/Users/msbabo/code/Graph-CoT/data/processed_data/biomedical/data.jsonl"
DEFAULT_GRAPH_JSON = "/Users/msbabo//code/Graph-CoT/data/processed_data/biomedical/graph.json"
DEFAULT_DB_URI     = 'mongodb://localhost:27017/'
DEFAULT_DB_NAME    = "graph-cot"
DEFAULT_DB_COLLECTION = "biomedical"
# --------------------------------------------------------------------|-------: 
def __handle_cli_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.2:1b", choices = MODELS) 
    parser.add_argument('--questions-jsonl', type=str, default=DEFAULT_QUES_JSON)
    parser.add_argument('-n', type=int, default =2, help="Number of questions")
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- |-- --- : 
    parser.add_argument("--dataset", type=str, default="biomedical")                      
    parser.add_argument("--openai_api_key", type=str, default="xxx")                
    parser.add_argument("--path", type=str)                                         
    parser.add_argument("--save_file", type=str)                                    
    parser.add_argument("--embedder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--faiss_gpu", type=bool, default=False)                    
    parser.add_argument("--embed_cache", type=bool, default=True)                   
    parser.add_argument("--max_steps", type=int, default=15)                        
    parser.add_argument("--zero_shot", type=bool, default=False)                    
    parser.add_argument("--graph_path", type=str, default="/Users/msbabo/code/Graph-CoT/data/processed_data/biomedical/graph.json")                    
    #parser.add_argument("--ref_dataset", type=str, default=None)                    
    # parser.add_argument("--llm_version", type=str ) --> model 
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- |-- --- : 
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
    

    # ^^^^ For local execution - we wont need this                              
    with open(args.questions_jsonl, 'r') as f:                                         
        contents = []                                                           
        for item in jsonlines.Reader(f):                                        
            contents.append(item)                                               
    print(f"Number of lines in questions: {len(contents)}")
                                                                                
    ####################################################################        
    # contents = [contents[80]]                                                 
    ####################################################################        
                                                                                
    output_file_path = "/Users/msbabo/code/mlhq-kg-mac/mlh/save_local"#qargs.save_file
                                                                                
    parent_folder = os.path.dirname(output_file_path)                           
    parent_parent_folder = os.path.dirname(parent_folder)                       
    if not os.path.exists(parent_parent_folder):                                
        os.mkdir(parent_parent_folder)                                          
    if not os.path.exists(parent_folder):                                       
        os.mkdir(parent_folder)                                                 
                                                                                
    if not os.path.exists('{}/logs'.format(parent_folder)):                     
        os.makedirs('{}/logs'.format(parent_folder))                            
    logs_dir = '{}/logs'.format(parent_folder) 
   
    if not args.zero_shot:                                                         
        agent_prompt = graph_agent_prompt                                          
    else:                                                                          
        agent_prompt = graph_agent_promtp_zeroshot 

    args.node_text_keys = args.node_text_keys = NODE_TEXT_KEYS["biomedical"] 
    args.embed_cache_dir = "/Users/msbabo/code/mlhq-kg-mac/mlhq/embed_cache_dir"
    agent = GraphAgent(
                max_steps = args.max_steps,
                args=args, # TODO - slowing unpack this 
                graph_path = args.graph_path, 
                model = args.model, 
                agent_prompt=agent_prompt, 
                data_domain = "biomedical", 
    )
    print(type(GraphAgent))
    print("DONE") 
    sys.exit(1)

    results = OrderedDict() 

    ollama_model = True if (args.model in OLLAMA_MODELS) else False 

    client = Client(model=args.model)
    sys_prompt = "Can you answer the following question in a concise manner: "

   
    for n, qdata in enumerate(args.questions, start=1): 
        if args.n > n: 
            break 
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
    #args.questions = load_jsonl(args.questions_jsonl) # qid, question, answer
    main(args)
