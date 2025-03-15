import os, sys, re, argparse, json, jsonlines
from tqdm import tqdm
import logging
import datetime
from IPython import embed
from collections import OrderedDict
from mlhq.backend.openai import Client 
from mlhq.backend.openai import MODELS, HF_MODELS, OLLAMA_MODELS
from mlhq.utils import proceed 
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
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
# --------------------------------------------------------------------|-------: 
def remove_fewshot(prompt: str) -> str:                                                                                                                                       
    # prefix = prompt.split('Here are some examples:')[0]                       
    # suffix = prompt.split('(END OF EXAMPLES)')[1]                             
    prefix = prompt[-1].content.split('Here are some examples:')[0]             
    suffix = prompt[-1].content.split('(END OF EXAMPLES)')[1]                   
    return prefix.strip('\n').strip() + '\n' +  suffix.strip('\n').strip()  
# --------------------------------------------------------------------|-------: 
def __handle_cli_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices = MODELS) 
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
    parser.add_argument("--zero-shot", type=bool, default=False)                    
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
#def proceed(): 
#    _proceed = input("\nContinue (y/n)? ")
#    if _proceed in ["Y", 'y']: return  
#    print("Exiting")
#    sys.exit(0)
# --------------------------------------------------------------------|-------: 
def main(args): 
    

    with open(args.questions_jsonl, 'r') as f:                                         
        contents = []                                                           
        for item in jsonlines.Reader(f):                                        
            contents.append(item)                                               
    print(f"Number of lines in questions: {len(contents)}")

    proceed() 
                                                                                
    ####################################################################        
    # contents = [contents[80]]                                                 
    ####################################################################        
                                                                                
    output_file_path = "/Users/msbabo/code/mlhq-kg-mac/mlhq/save_local"#qargs.save_file
    
    print(f"Output file path: {output_file_path}")
    #parent_folder = os.path.dirname(output_file_path)                           
    #parent_parent_folder = os.path.dirname(parent_folder)                       
    #if not os.path.exists(parent_parent_folder):                                
    #    os.mkdir(parent_parent_folder)                                          
    #if not os.path.exists(parent_folder):                                       
    #    os.mkdir(parent_folder)                                                 
    logs_dir = f'{output_file_path}/logs'
    if not os.path.exists(logs_dir):                     
        os.makedirs(logs_dir)     
    #proceed()    

    if not args.zero_shot:                                                         
        agent_prompt = graph_agent_prompt                                          
    else:                                                                          
        agent_prompt = graph_agent_prompt_zeroshot 

    args.node_text_keys = args.node_text_keys = NODE_TEXT_KEYS["biomedical"] 
    args.embed_cache_dir = "/Users/msbabo/code/mlhq-kg-mac/mlhq/embed_cache_dir"

    agent = GraphAgent(
                model = args.model,
                max_steps = args.max_steps,
                args=args, # TODO - slowing unpack this 
                graph_path = args.graph_path, 
                agent_prompt=agent_prompt, 
                data_domain = "biomedical", 
    )
    print(type(GraphAgent))
    #proceed() 


    unanswered_questions = []
    correct_logs = []
    halted_logs = []
    incorrect_logs = []
    generated_text = []

    for i in tqdm(range(len(contents))):
        ques = contents[i]['question']
        target = contents[i]['answer']
        print(f"{i}.) Question = {ques}")
        print(f"{i}.) Target   = {target}")
        proceed()
        agent.run(ques, target)
        print(f'Ground Truth Answer: {agent.key}')
        print('---------')
        log = f"Question: {contents[i]['question']}\n"
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n' if not args.zero_shot else agent._build_agent_prompt()[-1].content + f'\nCorrect answer: {agent.key}\n\n'
        with open(os.path.join(logs_dir, contents[i]['qid']+'.txt'), 'w') as f:
            f.write(log)

        ## summarize the logs
        if agent.is_correct():
            correct_logs.append(log)
        elif agent.is_halted():
            halted_logs.append(log)
        elif agent.is_finished() and not agent.is_correct():
            incorrect_logs.append(log)
        else:
            raise ValueError('Something went wrong!')

        generated_text.append({"question": contents[i]["question"], "model_answer": agent.answer, "gt_answer": contents[i]['answer']})
    
        if i <= 1: break 

    with jsonlines.open(output_file_path, 'w') as writer:
        for row in generated_text:
            writer.write(row)

    print(f'Finished Trial {len(contents)}, Correct: {len(correct_logs)}, Incorrect: {len(incorrect_logs)}, Halted: {len(halted_logs)}')
    print('Unanswered questions {}: {}'.format(len(unanswered_questions), unanswered_questions))





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
