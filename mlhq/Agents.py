from mlhq import Client
from mlhq.utils import panel_print, load_json, load_jsonl, proceed
import os 
import sys 
from datetime import datetime
import logging
import json
import argparse
from collections import OrderedDict
# ============================================================================:
class GraphInterface: 

    def __init__(self, graph_path): 
        
        # ....................................................................:
        self.graph = mlhq.utils.load_json(graph_path) 
        # Eventually, this needs to be a sophicated interface handling 
        # different DB backends 
        # ....................................................................:
    
    def __len__(self,): 
        return len(self.graph)

def load_text_prompt(file): 
    with open(file,"r") as fh: 
        data = fh.read().strip()
    return data
# ============================================================================:
# TODO: Make the Client class the inherited class: class QAClient(mlhq.Client)
class QAClient: 
    def __init__(self,
                 model=None, 
                 backend=None, 
                 token=None, 
                 system_prompt=None, 
                 prefix_prompt=None, 
                 gen_params=None, 
                 log_dir=None,
                 **kwargs): 
        self.client = Client(model=model,backend=backend,token=token)
        self._id = self.client.get_id()
        self.system_prompt = load_text_prompt(system_prompt)
        self.prefix_prompt = load_text_prompt(prefix_prompt)
        self.gen_params = gen_params 
        if log_dir: 
            if not os.path.exists(log_dir): 
                os.makedirs(log_dir)
                self.log_dir = logs_dir
        self.logger = logging.getLogger(f"{__name__}.GraphAgent") 
        self.logger.info(f"Created QAClient-{self._id}")
        self.logger.info(f"Logs can be found at: {self.log_dir}")
  
    def query(self, question): # TODO: Add gen_params 
        messages = []
        messages.append({'role':'system', 'content':self.system_prompt}) 
        if self.prefix_prompt: 
            messages.append({"role":'user', "content":self.prefix_prompt.format(question=question)})
        else: 
            messages.append({"role":'user', "content":question})

        response = self.client.text_generation(messages, **self.gen_params)

# ============================================================================:

# ============================================================================:
class GraphAgent:
    def __init__(self, 
                 graph_path=None, 
                 core_model=None, 
                 core_backend=None,
                 core_system_prompt=None,
                 core_gen_params=None,
                 judge_model=None, 
                 judge_backend=None,
                 judge_system_prompt=None,
                 judge_prefix_prompt=None,
                 judge_gen_params=None,
                 config_path=None, 
                 **kwargs):
        """
        Initialize a GraphAgent with model configurations.
        
        Args can be passed directly or through a config dictionary using **config.
        """
        # ....................................................................:
        self.logger = logging.getLogger(f"{__name__}.GraphAgent") 
        self.logger.debug("Creating GraphAgent")
        self._core_llm = None
        self._judge_llm = None
        self._graph = GraphInterface(graph_path)
        self.config_path = config_path
        # ....................................................................:
        self._core_llm = mlhq.Client(core_model, core_backend)
        self._judge_llm = mlhq.Client(judge_model, judge_backend)
        self._core_llm_id = self._core_llm._id #id(self._core_llm)
        self._judge_llm_id = self._judge_llm._id #id(self._judge_llm)

        self._core_gen_params = core_gen_params
        self._judge_gen_params = judge_gen_params
        # TODO: We need to pass the sys-prompt and prefix-prompts to 
        #       to the client agent. 
        #self._core_llm.set_system_prompt(load_text_prompt(core_system_prompt))
        self._core_system_prompt = load_text_prompt(core_system_prompt)
        self._judge_system_prompt = load_text_prompt(judge_system_prompt)
        self._judge_prefix_prompt = load_text_prompt(judge_prefix_prompt)
        # ....................................................................:
        self._enable_json_logging = True                                                                                               
        self._log_dir = os.path.join(os.getcwd(), "ga-logs")                       
        os.makedirs(self._log_dir, exist_ok=True)                                             
        self._cl_log = os.path.join(self._log_dir, f"MLHQ-GA-core-llm-{self._core_llm_id}-{datetime.now().strftime('%Y%m%d')}.jsonl")
        self._jl_log = os.path.join(self._log_dir, f"MLHQ-GA-judge-llm-{self._judge_llm_id}-{datetime.now().strftime('%Y%m%d')}.jsonl")
        # ....................................................................:
        self._save_interactions = True 
        self._inter_dir = os.path.join(os.getcwd(), "ga-interaction-logs") 
        os.makedirs(self._inter_dir, exist_ok=True)                                             
        self._inter_log = os.path.join(self._inter_dir, f"mlhq-judge-llm-{self._judge_llm_id}-{datetime.now().strftime('%Y%m%d')}.jsonl")
        # {
        #   'core_llm' : {
        #       'model' : <model>
        #       'gen-params' : gen-params, 
        #       'messages' : messages, 
        #   }
        #   'judge_llm' : {
        #       'model' : <model>
        #       'gen-params' : gen-params, 
        #       'messages' : messages, 
        #   }
        # }


    def _query_with_judgement(self, query): 
        messages = []
        if self._core_system_prompt: 
            messages.append({'role':'system', 'content':self._core_system_prompt}) 
        messages.append({"role":'user', "content":query})
        response = self._core_llm.text_generation(messages, **self._core_gen_params)

        panel_print(tag=f"Question", content=query, fc="red", border_style="yellow")
        panel_print(tag=f"Response-{self._core_llm._id}", content=response, fc="red", border_style="yellow")
        change_response = input("\nDo you wish to change the LLMs response? (y/n)")
        if change_response.upper() == "Y": 
            response = input("\n>>>")
            print(f"\n{response}")
        messages.append({'role':'assistant','content':response})

        jmessages = []    
        jmessages.append({'role':'system', 'content':self._judge_system_prompt}) 
        jmessages.append({'role':'user', 'content':f"{self._judge_prefix_prompt.format(interaction=_cleanup_messages(messages))}"})
        judgement = self._judge_llm.text_generation(jmessages, **self._judge_gen_params)
        jmessages.append({'role':'assistant','content':judgement})

        with open(self._cl_log, 'a') as f:
            for m in messages: 
                f.write(json.dumps(m) + '\n')
        with open(self._jl_log, 'a') as f:
            for m in jmessages: 
                f.write(json.dumps(m) + '\n')
   
        cl_tokens  = len(self._core_llm.tokenizer.apply_chat_template(                                                                           
                      messages,                                                   
                      tokenize=True,                                                              
                      add_generation_prompt=True))
        jl_tokens  = len(self._judge_llm.tokenizer.apply_chat_template(                                                                           
                      jmessages,                                                   
                      tokenize=True,                                                              
                      add_generation_prompt=True))
        self.logger.info(f"Total core-llm tokens used: {cl_tokens}")
        self.logger.info(f"Total judge-llm tokens used: {jl_tokens}")

        panel_print(tag="Question", content=question)
        panel_print(tag=f"Response-{self._core_llm._id}", content=response)
        panel_print(tag=f"Judgement-{self._judge_llm._id}", content=judgement)
        
        save_case = input("\n>>> Do you want to save this interaction? (y/n)")
        if save_case.upper() in ["Y", "YES"]: 
            category = input(">>> Was this Good or Bad? ")
            note     = input(">>> Quick description: ")
            json_log = OrderedDict()
            json_log['classification'] = 'good' if category.upper() in ["G","GOOD"] else 'bad'
            json_log['note'] = note
            json_log['config_path'] = self.config_path
            json_log['core_llm'] = {'model':self._core_llm.get_model_name(), 
                                    'backend': self._core_llm.backend, 
                                    'gen_params': self._core_gen_params,
                                    'messages' : messages}

            json_log['judge_llm'] = {'model':self._judge_llm.get_model_name(), 
                                    'backend': self._judge_llm.backend, 
                                    'gen_params': self._judge_gen_params,
                                    'messages' : jmessages}
            with open(self._inter_log, 'w') as fp:
                json.dump(json_log, fp, indent=4)
            self.logger.info(f"Log interaction at: {self._inter_log}")
        return response, judgement

    @property
    def graph(self): 
        return self._graph
# ============================================================================:
class JudgeClient: 
    def __init__(self, 
                 model=None,                                                       
                 backend=None,                                                     
                 token=None,                                                       
                 system_prompt=None,                                               
                 prefix_prompt=None,                                               
                 gen_params=None,                                                  
                 log_dir=None,                                                     
                 **kwargs):                                                        

        self.client = Client(model=model,backend=backend,token=token)              
        self._id = self.client.get_id()                                            
        self.system_prompt = load_text_prompt(system_prompt)                       
        self.prefix_prompt = load_text_prompt(prefix_prompt)                       
        self.gen_params = gen_params                                               
        if not log_dir:                                                            
            log_dir = os.path.join(os.get_cwd(), "logs", "BasicClient")            
        if not os.path.exists(log_dir):                                            
            os.makedirs(log_dir, exist_ok=True)                                    
        self.log_dir = log_dir                                                     
                                                                                   
        now = datetime.now()                                                       
        timestamp = now.strftime("%Y%m%d_%H%M%S")                                  
        self.log_file = os.path.join(self.log_dir, f"basicclient-{self._id}-{timestamp}.jsonl")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}") 
        self.logger.info(f"Created {self.__class__.__name__}-{self._id}")          
        self.logger.info(f"Logs can be found at: {self.log_dir}")                  
        self.logger.info(f"Specific log file: {self.log_file}") 

    def get_id(self): 
        return self._id
    def get_log_file(self): 
        return self.log_file

  
    def judge(self, question, answer, ground_truth): 
        messages = [
            {'role':'system', 'content':self.system_prompt}, 
            {'role':'user', 'content':self.prefix_prompt.format(
                question=question,
                llm_answer = answer,
                ground_truth=ground_truth)},
        ]
        judgement = self.client.text_generation(messages, **self.gen_params) 
        messages.append({'role':'assistant','content':judgement}) 

        if self.log_dir and self.log_file:                                      
            with open(self.log_file, 'a') as f:                                 
                for m in messages:                                              
                    f.write(json.dumps(m) + '\n')  
        return judgement 

# ============================================================================:
def _get_qa(messages): 
    q = "" 
    a = "" 
    assert len(messages) in [2,3], "Messages length too long"
    for m in messages: 
        if   m['role'] == "user":      q = m['content']
        elif m['role'] == "assistant": a = m['content']
        elif m['role'] == 'system':    continue  
    return q, a 

def _cleanup_messages(messages): 
    rstr = []
    for m in messages: 
        if m['role'] == "user": 
            rstr.append(f"User: {m['content']}")
        elif m['role'] == "assistant": 
            rstr.append(f"LLM: {m['content']}")
        elif m['role'] == 'system': 
            continue  
        else: 
            raise RuntimeError("Bad role: {m['role']}")
    return "\n".join(rstr)
# ============================================================================:
class BasicClient: 
    def __init__(self,
                 model=None, 
                 backend=None, 
                 token=None, 
                 system_prompt=None, 
                 prefix_prompt=None, 
                 gen_params=None, 
                 log_dir=None,
                 **kwargs): 
        self.client = Client(model=model,backend=backend,token=token)
        self._id = self.client.get_id()
        self.system_prompt = load_text_prompt(system_prompt)
        self.prefix_prompt = load_text_prompt(prefix_prompt)
        self.gen_params = gen_params 
        if not log_dir: 
            log_dir = os.path.join(os.get_cwd(), "logs", "BasicClient")
        if not os.path.exists(log_dir): 
            os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"basicclient-{self._id}-{timestamp}.jsonl")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}") 
        self.logger.info(f"Created {self.__class__.__name__}-{self._id}")
        self.logger.info(f"Logs can be found at: {self.log_dir}")
        self.logger.info(f"Specific log file: {self.log_file}")

    def get_id(self): 
        return self._id
    def get_log_file(self): 
        return self.log_file
  
    def query(self, question): # TODO: Add gen_params 
        messages = []
        messages.append({'role':'system', 'content':self.system_prompt}) 
        messages.append({"role":'user', "content":self.prefix_prompt.format(question=question)})
        response = self.client.text_generation(messages, **self.gen_params)
        messages.append({'role':'assistant', 'content':response})
        if self.log_dir and self.log_file: 
            with open(self.log_file, 'a') as f:
                for m in messages: 
                    f.write(json.dumps(m) + '\n')
        return response

# ============================================================================:
def __handle_cli_args(): 
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str,required=True)
    parser.add_argument('--question', type=str, default="What is the capital of California?")
    parser.add_argument('--use-graph-questions', action="store_true")
    parser.add_argument('--domain', type=str, default="biomedical")
    args = parser.parse_args()
    return args 
# ============================================================================:
if __name__ == "__main__": 
    args = __handle_cli_args()
    question = args.question
    #.........................................................................:
    now = datetime.now()                                                       
    timestamp = now.strftime("%Y%m%d_%H%M%S")                                  

    logs_dir = os.path.join(os.getcwd(), "logs", "e2e-baseline") 
    classifier_logs_dir = os.path.join(os.getcwd(), "logs", "e2e-baseline", "classifier") 
    base_logs_dir = os.path.join(os.getcwd(), "logs", "e2e-baseline", "base") 
    judge_logs_dir = os.path.join(os.getcwd(), "logs", "e2e-baseline", "judge") 
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(classifier_logs_dir, exist_ok=True)
    os.makedirs(base_logs_dir, exist_ok=True)
    os.makedirs(judge_logs_dir, exist_ok=True)
    e2e_log_file = os.path.join(logs_dir, f"e2e-{timestamp}.jsonl")
    #.........................................................................:
    #config = load_json(args.config)
    classifier_config = {
        "model"         : "meta-llama/Llama-3.1-8B-Instruct", 
        "backend"       : "hf-local", 
        "system_prompt" : "/Users/msbabo/code/mlhq-kg-mac/mlhq/system-prompts/msee/llm-classifier-system-prompt-1.txt",  
        "prefix_prompt" : "/Users/msbabo/code/mlhq-kg-mac/mlhq/system-prompts/msee/llm-classifier-prefix-prompt-1.txt", 
        "log_dir"       : classifier_logs_dir, 
        "gen_params"    : {"temperature":0.1, "max_new_tokens":200}
    }
    qa_config = {
        "model"         : "meta-llama/Llama-3.1-8B-Instruct", 
        "backend"       : "hf-local", 
        "system_prompt" : "/Users/msbabo/code/mlhq-kg-mac/mlhq/system-prompts/msee/qa-system-prompt-1.txt",  
        "prefix_prompt" : "/Users/msbabo/code/mlhq-kg-mac/mlhq/system-prompts/msee/qa-prefix-prompt-1.txt", 
        "log_dir"       : base_logs_dir, 
        "gen_params"    : {"temperature":0.1, "max_new_tokens":320}
    }
    judge_config = {
        "model"         : "meta-llama/Llama-3.3-70B-Instruct", 
        "backend"       : "hf-client", 
        "system_prompt" : "/Users/msbabo/code/mlhq-kg-mac/mlhq/system-prompts/msee/final-judge-system-prompt-msee-1.txt", 
        "prefix_prompt" : "/Users/msbabo/code/mlhq-kg-mac/mlhq/system-prompts/msee/final-judge-prefix-prompt-msee-1.txt", 
        "log_dir"       : judge_logs_dir, 
        "gen_params"    : {"temperature":0.1, "max_new_tokens":320}
    }
    #.........................................................................:
    domain = 'test'
    qa_paths = {
        'test' : [
            {'qid':0, 'question':"What is the capital of California?",'answer':"Sacramento"}, 
            {'qid':1, 'question':"How many states are in the US?",'answer':"50"}, 
        ]
    }
    # TODO: We should add to the JUDGELLM system/prefix prompts that the ground truth 
    # answers could simply be a list of keywords. 
 
    # ........................................................................:
    RouteLLM = BasicClient(**classifier_config)
    QALLM    = BasicClient(**qa_config)
    JudgeLLM = JudgeClient(**judge_config)

    id_keys = {'RouteLLM':RouteLLM.get_id(),'QALLM' : QALLM.get_id(),'JudgeLLM' : JudgeLLM.get_id()}
    with open(e2e_log_file, 'a') as f:   
        f.write(json.dumps(id_keys) + '\n') 
    # ........................................................................:
    
    # ........................................................................:
    #qa_data = load_jsonl(qa_paths[domain])                                      
    qa_data = qa_paths[domain]  
    for i, qdata in enumerate(qa_data):                                         
        question = qdata['question']                                            
        ground_truth  = qdata['answer']                                              
        qid      = qdata['qid']  
        
        classification = RouteLLM.query(question)
        # If internal --> pass to QALLM 
        # If External --> GraphAgent 
        response       = QALLM.query(question)
        judgement      = JudgeLLM.judge(question , response, ground_truth)

        panel_print(tag="Question", content=question)
        panel_print(tag=f"RouteLLM-{RouteLLM.get_id()}", content=classification)
        panel_print(tag=f"QALLM-{QALLM.get_id()}", content=response)
        panel_print(tag=f"JudgeLLM-{JudgeLLM.get_id()}", content=judgement)

        proceed()
    
    # ........................................................................:
    sys.exit(0)

    print(f"DEBUG: Dumping incoming config: {os.path.abspath(args.config)}")
    for k,v in config.items(): 
        print("DEBUG: %24s --> %s"%(k,v))
    #.........................................................................:
    
    domain = args.domain 
    qa_paths = {                                                                       
        "biomedical" : "/Users/msbabo/code/Graph-CoT/data/processed_data/biomedical/data.jsonl",
        "amazon" : "/Users/msbabo/code/Graph-CoT/data/processed_data/amazon/data.jsonl",
        "dblp" : "/Users/msbabo/code/Graph-CoT/data/processed_data/dblp/data.jsonl",
        "goodreads": "/Users/msbabo/code/Graph-CoT/data/processed_data/goodreads/data.jsonl",
        "legal": "/Users/msbabo/code/Graph-CoT/data/processed_data/legal/data.jsonl",
        "maple-biology": "/Users/msbabo/code/Graph-CoT/data/processed_data/maple/Biology/data.jsonl",
        "maple-medicine": "/Users/msbabo/code/Graph-CoT/data/processed_data/maple/Medicine/data.jsonl",
        "maple-physics": "/Users/msbabo/code/Graph-CoT/data/processed_data/maple/Physics/data.jsonl",
        "maple-materials-science": "/Users/msbabo/code/Graph-CoT/data/processed_data/maple/Materials_Science/data.jsonl",
        "maple-chemistry": "/Users/msbabo/code/Graph-CoT/data/processed_data/maple/Chemistry/data.jsonl", 
    }
    qa_data = load_jsonl(qa_paths[domain])
    for i, qdata in enumerate(qa_data):                                         
        question = qdata['question']                                            
        answer   = qdata['answer']                                              
        qid      = qdata['qid']  
        
        response, judgement = ga._query_with_judgement(question)

        panel_print(tag="Question", content=question)
        panel_print(tag=f"Response-{ga._core_llm._id}", content=response)
        panel_print(tag=f"Judgement-{ga._judge_llm._id}", content=judgement)
        panel_print(tag="Answer", content=answer)

        #gen_params = {"max_new_tokens":256, "temperature":0.5}
        #gen_params = {"max_new_tokens":1024, "temperature":0.0, "top_p":1.0, "top_k":0, "do_sample":False}
        gen_params = {"max_new_tokens":512, "do_sample":False}
        ja = final_judge.judge(question, response, answer, gen_params)
        print(ja)
        panel_print(tag="Final-Judge", content=ja, border_style='cyan')
        proceed()
     
        
