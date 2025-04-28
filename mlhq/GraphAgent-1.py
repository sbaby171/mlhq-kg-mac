from mlhq import Client
from mlhq.utils import load_json, load_jsonl, panel_print, proceed
from tools.retriever import Retriever
import argparse
import sys
import time 
import collections 
import os
import re 
import json 
import string
import torch 
import gc 
# -----------------------
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from IPython import embed
from textwrap import dedent
# ============================================================================:
# ============================================================================:
# ============================================================================:
def get_size(obj, seen=None):
    """Recursively find the size of an object and its contents in bytes."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important: Mark this object as seen
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, collections.deque)):
        size += sum(get_size(item, seen) for item in obj)
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__slots__'):
        size += sum(get_size(getattr(obj, slot), seen) for slot in obj.__slots__ if hasattr(obj, slot))
        
    return size

def format_size(size_bytes):
    """Format the size in a human-readable format."""
    for unit in ['', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size_bytes < 1024.0 or unit == 'PiB':
            if unit == '':
                return f"{size_bytes} bytes"
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0

def dict_memory_usage(dictionary):
    """Calculate and return the memory usage of a dictionary in human-readable format."""
    memory_bytes = get_size(dictionary)
    return format_size(memory_bytes)

def split_checks(input_string):                                                 
    pattern = r'\w+\[.*?\]'                                                     
    # Use re.findall to get all matches                                         
    result = re.findall(pattern, input_string)                                  
    return result                                                               
def remove_quotes(s):                                                           
    if s.startswith(("'", '"')) and s.endswith(("'", '"')):                     
        return s[1:-1]                                                          
    return s                                                                    
def get_action_list(string): 
    if string[:len('Finish')] == 'Finish':                                      
        return [string]                                                         
    else:                                                                       
        # return string.split(', ')                                             
        return split_checks(string) 

def parse_action(string): 
    print(f'DEBUG: [parse_action] --> Incoming string:{string}')                  
    pattern = r'^(\w+)\[(.+)\]$'                                                
    match = re.match(pattern, string)                                           
    if match:                                                                   
        action_type = match.group(1)                                            
        argument = match.group(2)                                               
        return action_type, argument                                            
    else:                                                                       
        return None 

def normalize_answer(s):                                                           
  def remove_articles(text):                                                       
    return re.sub(r"\b(a|an|the|usd)\b", " ", text)                                
                                                                                   
  def white_space_fix(text):                                                       
      return " ".join(text.split())                                                
                                                                                   
  def remove_punc(text):                                                           
      exclude = set(string.punctuation)                                            
      return "".join(ch for ch in text if ch not in exclude)                       
                                                                                   
  def lower(text):                                                                 
      return text.lower()                                                          
                                                                                   

  return white_space_fix(remove_articles(remove_punc(lower(s))))                   
  #return white_space_fix(remove_articles(remove_punc(s.lower())))    
                                                                                   
def EM(answer, key) -> bool:                                                       
    return check_strings_llm(answer, key)
    norm_ans = normalize_answer(str(answer)) 
    norm_key = normalize_anser(str(key))
    #return normalize_answer(str(answer)) == normalize_answer(str(key)) 
    if norm_ans == norm_key:
        return True  
    else: 
        return False 
    #return normalize_answer(str(answer)) == normalize_answer(str(key)) 
# ============================================================================:
# ============================================================================:
# ============================================================================:
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
graph_paths = {
    "biomedical": "/Users/msbabo/code/Graph-CoT/data/processed_data/biomedical/graph.json", 
    "maple-materials-science" : "/Users/msbabo/code/Graph-CoT/data/processed_data/maple/Materials_Science/graph.json", 
}
# SYSTEM_PROMPT.format(examples=examples)
SYSTEM_PROMPT = """
You are GraphAgent, an LLM that answers questions through explicit step-by-step reasoning and calling graph tools(aka actions). The steps are: 
- **Thought** (what you are thinking, why you are taking an action)
- **Action** (interaction with graph through one of the allowed operations below)
- **Observation** (the result from your action)

Allowed actions (choose according to your reasoning):

(1) **RetrieveNode[keyword]**:  
Retrieve the node from the graph best matching the keyword or query.

(2) **NodeFeature[Node, feature]**:  
Get the value of a specific attribute ('feature') of a given node ('Node'). Use attributes such as names, descriptions, or identifiers like inchikeys. Do not return internal IDs unless explicitly necessary.

(3) **NodeDegree[Node, neighbor_type]**:  
Count the number of neighbors of the specified type ('neighbor_type') connected to the node ('Node').

(4) **NeighbourCheck[Node, neighbor_type]**:  
List neighbors of a specified type ('neighbor_type') connected to the node ('Node').

Guidelines for Answering:
- For questions asking "how many" or numeric questions, explicitly use NodeDegree.
- For questions requiring names, descriptions, or unique identifiers, explicitly retrieve node features with NodeFeature.
- For yes/no questions, check relevant neighbors with NeighbourCheck or NodeDegree and clearly state your reasoning and final answer explicitly as "Yes" or "No".

Take as many steps as necessary, clearly documenting your thought process.

## Examples -  demonstrating reasoning clearly:

{examples}

(END OF EXAMPLES)
"""
EXAMPLES = {
    "biomedical" :  dedent(""" 
        ### Example-1:
        Question: What compounds can be used to treat Crohn's disease? Please answer the compound names rather than IDs.
        Thought 1: The question is related to a disease node (Crohn's disease). We need to find the node in the graph.
        Action 1: RetrieveNode[Crohn's disease]
        Observation 1: The ID of this node is DOID:8778.
        Thought 2: The question is asking the compounds which can be used to treat a disease, we need to check the node's 'Compound-treats-Disease' neighbor from the graph.
        Action 2: NeighbourCheck[DOID:8778, Compound-treats-Disease]
        Observation 2: ['DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033']
        Thought 3: The IDs of the compounds are 'DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033'. We need to get their names.
        Action 3: NodeFeature[DB01014, name], NodeFeature[DB00244, name], NodeFeature[DB00795, name], NodeFeature[DB00993, name], NodeFeature[DB00635, name], NodeFeature[DB01033, name]
        Observation 3: Balsalazide, Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine
        Thought 4: The name of compounds are Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Action 4: Finish[Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine]
        
        ### Example-2:
        Question: What is the inchikey of Caffeine?
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph.
        Action 1: RetrieveNode[Caffeine]
        Observation 1: The ID of this node is DB00201.
        Thought 2: The question is asking the inchikey feature of a node, we need to check the node's 'inchikey' feature from the graph.
        Action 2: NodeFeature[DB00201, inchikey]
        Observation 2: InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N
        Thought 3: The inchikey of the node is InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Action 3: Finish[InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N]
        
        ### Example-3:
        Question: How many side effects does Caffeine have?
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph.
        Action 1: RetrieveNode[Caffeine]
        Observation 1: The ID of this node is DB00201.
        Thought 2: The question is asking the number of side effects a compound has, we need to calculate the number of the node's 'Compound-causes-Side Effect' neighbors from the graph.
        Action 2: NodeDegree[DB00201, 'Compound-causes-Side Effect']
        Observation 2: 58
        Thought 3: The number of 'Compound-causes-Side Effect' neighbors are 58.
        Action 3: Finish[58]"""),

    "maple": dedent("""
        ### Example-1: 
        Question: When was the paper Strongly Interacting Higgs Sector in the Minimal Standard Model published?
        Thought 1: The question is asking some basic information of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). We need to find the node in the graph.
        Action 1: RetrieveNode[Strongly Interacting Higgs Sector in the Minimal Standard Model]
        Observation 1: The ID of this node is 3101448248.
        Thought 2: The question is asking the published date of a paper, we need to check the node feature (year) from the graph.
        Action 2: NodeFeature[3101448248, year]
        Observation 2: 1993
        Thought 3: The published date of the paper is 1993.
        Action 3: Finish[1993]

        ### Example-2: 
        Question: How many authors do the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars have?
        Thought 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.
        Action 1: RetrieveNode[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Thought 2: The question is asking the number of authors of a paper, we need to calculate the node's author neighbor degree from the graph.
        Action 2: NodeDegree[2090642949, author]
        Observation 2: 2
        Thought 3: The number of the authors is 2
        Action 3: Finish[2]

        ### Example-3: 
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Thought 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.
        Action 1: RetrieveNode[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Thought 2: The question is asking the published venue of a paper, we need to check the node's venue neighbor from the graph.
        Action 2: NeighbourCheck[2090642949, venue]
        Observation 2: ['1980519', '1053242']
        Thought 3: The ID of the published venue are 1980519 and 1053242. We need to get their names.
        Action 3: NodeFeature[1980519, name], NodeFeature[1053242, name]
        Observation 3: the astrophysical journal, the atmosphere journal
        Thought 4: The name of the published venues are the astrophysical journal and the atmosphere journal
        Action 4: Finish[the astrophysical journal, the atmosphere journal]"""), 

} 

GRAPH_DEF = {
    "biomedical" : """There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.\nEach node has name feature.\nThere are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.""", 

    "maple": """Definition of the graph: There are three types of nodes in the graph: paper, author and venue.                       
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes."""
} 


# PREFIX_PROMPT.format(graph_definition = graph_definition, question=question, scratchpad=scratchpad)
PREFIX_PROMPT ="""
Graph definition:
{graph_definition}

Now, carefully answer the following question. Use human-readable node attributes (such as names or labels) rather than internal IDs wherever possible.

Question: {question}

{scratchpad}
"""

GRAPH_DEFINITION = json.dumps({
    "node_types": [
        "Anatomy", "Biological Process", "Cellular Component", "Compound",
        "Disease", "Gene", "Molecular Function", "Pathway",
        "Pharmacologic Class", "Side Effect", "Symptom"
    ],
    "node_features": ["name"],
    "edge_types": [
        "Anatomy-downregulates-Gene", "Anatomy-expresses-Gene",
        "Anatomy-upregulates-Gene", "Compound-binds-Gene",
        "Compound-causes-Side Effect", "Compound-downregulates-Gene",
        "Compound-palliates-Disease", "Compound-resembles-Compound",
        "Compound-treats-Disease", "Compound-upregulates-Gene",
        "Disease-associates-Gene", "Disease-downregulates-Gene",
        "Disease-localizes-Anatomy", "Disease-presents-Symptom",
        "Disease-resembles-Disease", "Disease-upregulates-Gene",
        "Gene-covaries-Gene", "Gene-interacts-Gene",
        "Gene-participates-Biological Process",
        "Gene-participates-Cellular Component",
        "Gene-participates-Molecular Function", "Gene-participates-Pathway",
        "Gene-regulates-Gene", "Pharmacologic Class-includes-Compound"
    ]
}, indent=2)

# ============================================================================:
# ============================================================================:
# ============================================================================:
# NOTE: Most of this was integrated from ./tools/graph_funcs
class GraphInterface:                                                              
                                                                                   
    def __init__(self, graph_path):                                                
        self.graph = load_json(graph_path)                              
        self._reset(self.graph) # Set graph_index

    def __len__(self,):                                                            
        return len(self.graph)  

    def get_mem_size(self,): 
        return dict_memory_usage(self.graph)

    def _reset(self, graph):
        graph_index = {}
        nid_set = set()
        for node_type in graph:
            for nid in graph[node_type]:
                assert nid not in nid_set
                nid_set.add(nid)
                graph_index[nid] = graph[node_type][nid]
        self.graph_index = graph_index

    def check_neighbours(self, node, neighbor_type=None):
        if neighbor_type:
            return str(self.graph_index[node]['neighbors'][neighbor_type])
        else:
            return str(self.graph_index[node]['neighbors'])

    def check_nodes(self, node, feature=None):
        """check the attributes of the nodes"""
        if feature:
            return str(self.graph_index[node]['features'][feature])
        else:
            return str(self.graph_index[node]['features'])

    def check_degree(self, node, neighbor_type):
        return str(len(self.graph_index[node]['neighbors'][neighbor_type]))
# ============================================================================:
class GraphAgent: 
    def __init__(self, 
                model = None,
                backend = None, 
                domain = None, 
                graph_path = None, 
                system_prompt = None, 
                prefix_prompt = None,
                examples = None, 
                graph_definition = None, 
                gen_params = None, 
                embedder_model = None, 
                steps = 10,
                **kwargs):

        self.graph_path = graph_path
        self.embedder_model = embedder_model
        self.domain = domain 

        self.system_prompt = system_prompt
        self.examples = examples 
        self.prefix_prompt = prefix_prompt
        self.graph_definition = graph_definition
        self.gen_params = gen_params

        self.graph = GraphInterface(self.graph_path) 
        print(f"DEBUG: [GraphAgent.init]: Graph memory-usage: {self.graph.get_mem_size()}")
        embed_cache_base = "/Users/msbabo/code/mlhq-kg-mac/mlhq/embed_cache_dir/"
        embed_cache_dir = os.path.join(embed_cache_base, domain)
        print(f"DEBUG: [GraphAgent.init] Domain = {domain}")
        print(f"DEBUG: [GraphAgent.init] Embed Cache Directory = {embed_cache_dir}")
        if not os.path.exists(embed_cache_dir): 
            raise RuntimeError(f"Invalid embedded cache directory: {embed_cache_dir}")
        # cache-all-mpnet-base-v2.pkl
        self.embed_cache_dir = embed_cache_dir
        self.retriever = Retriever(self.graph.graph, self.domain, self.embedder_model, cache_dir = self.embed_cache_dir) 
        # ^^^ TODO: Integrate GraphInterface and Retriever
        self.client = Client(model=model, backend=backend)
        self.finished = False
        self._correct = -1 
        self.scratchpad = [] 
        self.steps = steps 
   
    def reset(self):
        self.finished = False
        self._correct = - 1 
        self.scratchpad = [] 
        

    def is_finished(self) -> bool:                                              
        return self.finished 

    def is_correct(self, answer, ground_truth):
        return self.check_strings_llm(answer, ground_truth)

    def evaluate(self, question, ground_truth): 
        func = "GA.evaluate"
        messages = [
            { 'role':'system', 
              'content':self.system_prompt.format(examples=self.examples)
            }, 
            { 
              "role":'user', 
              "content":self.prefix_prompt.format(
                            graph_definition=self.graph_definition,
                            question=question, 
                            scratchpad="\n".join(self.scratchpad))
            }
        ]
        response = self.client.text_generation(messages, **self.gen_params) 

        RE_THOUGHT = re.compile("Thought\s+(?P<num>\d+)\s*:\s*(?P<thought>.*)")
        RE_ACTION = re.compile("Action\s+(?P<num>\d+)\s*:\s*(?P<action>.*)")

        keep_lines = [] 
        rlines = response.split("\n")
        
        print("\nDumping Response line-by-line: ")
        print(  "------------------------------")
        for rl in rlines: 
            rl = rl.strip()
            if (not rl) or (rl == ""): continue 
            if self.is_finished(): continue
            print(f"RL: {rl}\n")

            keep_lines.append(rl) # ------ NOTE - We start w. KEEPING LINE 
            if rl.startswith("Action"): 
                m = RE_ACTION.search(rl)
                if m: 
                    num = m.group("num")
                    action = m.group("action")
                    print(f"DEBUG: [{func}]: Action -- num={num}, action={action}")
                else: 
                    raise RuntimeError("Bad Action parsing: {rl}")
                observation = f"Observation {num}: "
                action_list = get_action_list(action)
                node_feature_count = 0
                for cnt, tmp_action in enumerate(action_list, start=0):
                    try:                                                         
                        action_type, argument = parse_action(tmp_action) 
                    except:                                                      
                        err_cnt += 1                                             
                        err_msg= 'There is something wrong with the generated target actions.'
                        #self._repsonses[err_key.format(err_cnt)] = {'raw':'','trimmed':mesg, 'accounted':False}
                        raise RuntimeError(f"Bad Action parsing: {action}")

                    print(f"DEBUG: [{func}]: Action={action_type}, args={argument}")    
                   
                    if action_type == 'RetrieveNode': # ************************* RETRIEVENODE
                        try:                                                     
                            idd, node = self.retriever.search_single(argument, 1)
                            observation += f"The ID of this retrieval target node is {idd}."
                        except: 
                            raise RuntimeError(f"Bad RetrieveNode call ")
    
                    elif action_type == 'NeighbourCheck': # ********************* NEIGHBOURCHECK
                        node_id, neighbor_type = argument.split(', ')            
                        node_id = remove_quotes(node_id)                         
                        neighbor_type = remove_quotes(neighbor_type)             
                        x = (self.graph.check_neighbours(node_id,neighbor_type))
                        observation += x

                    elif action_type == 'NodeFeature': # ************************ NODEFEATURE
                        node_id, feature_name = argument.split(', ')                   
                        node_id = remove_quotes(node_id)                               
                        feature_name = remove_quotes(feature_name)                     
                        print(f"DEBUG: [{func}]: NodeFeature: node-id={node_id}")
                        print(f"DEBUG: [{func}]: NodeFeature: feature-name={feature_name}")
                        print(f"DEBUG: [{func}]: --> len(action_list) = {len(action_list)}")
  
                        if feature_name.startswith("participates") and self.domain == "biomedical": 
                            feature_name = "Gene-{feature_name}"
                            print(f"DEBUG: [{func}]: Updated: NodeFeature: feature-name={feature_name}")


                        entry = self.graph.check_nodes(node_id, feature_name)
                        print(f"DEBUG: [{func}]: Entry = {entry}, node-feature-count = {node_feature_count}")
                        if node_feature_count == 0 :                             
                            observation += entry
                        else:
                            observation += ", " + entry
                        node_feature_count += 1                              

                    elif action_type == 'Finish':# ******************************* FINISH 
                        try:                                                     
                            answer = eval(argument)                         
                        except:                                                  
                            answer = argument                               
                        print(f"DEBUG: [{func}]:       Answer = {answer}")
                        print(f"DEBUG: [{func}]: Ground-Truth = {ground_truth}")
                        if self.is_correct( answer, ground_truth):                                    
                            observation += 'Answer is CORRECT'                   
                            self._correct = 1
                        else:                                                    
                            observation += 'Answer is INCORRECT'                 
                            self._correct = 0
                        self.finished = True                                     

                panel_print(tag="Observation", content=observation)
                keep_lines.append(observation)
    
            elif rl.startswith("Thought"): 
                m = RE_THOUGHT.search(rl)
                if m: 
                    num = m.group("num")
                    thought = m.group("thought")
                    print(f"DEBUG: [{func}]: Thought -- num={num}, thought={thought}")
                else: 
                    raise RuntimeError("Bad Action parsing: {rl}")

            else: 
                print(f"DEBUG: [{func}]: Noisy line: {rl}")
                del keep_lines[-1] 
                # NOTE: we need to add the lines at the start because ow . we would 
                #       be adding the observation before we add the action
                #       So, we because we add at the start, we need to make sure to 
                #       delete before leaving.
        self.scratchpad.extend(keep_lines)

        return response, "\n".join(keep_lines)

    def check_strings_llm(self, string1, string2): 
        sys_prompt = dedent("""                                                            
You will be given two comma-separated strings called **string1** and **string2**.
                                                                                
• Treat each string as an unordered set of items.                               
• Ignore leading/trailing spaces and letter-case when comparing items.          
• If the two sets contain exactly the same items, respond with the single token:
                                                                                
    STRINGS_SAME                                                                
                                                                                
• Otherwise respond with the single token:                                      
                                                                                
    STRINGS_DIFFERENT                                                           
                                                                                
Respond with **only** one of those tokens—no extra text, punctuation, or explanation.
                                                                                
Example                                                                         
string1 = Sensitization, Burns second degree, Chemical burn, Chemical injury, Conjunctivitis  
string2 = Conjunctivitis, Sensitization, Chemical burn, Burns second degree, Chemical injury  
--> STRINGS_SAME                                                                
        """)                                                         
        x = f"\nstring1='{string1}',\nstring2='{string2}'"                          
        messages = []                                                               
        messages.append({"role": "system", "content":sys_prompt})                   
        messages.append({"role": "user", "content": x})                             
        resp = self.client.text_generation(messages)                              
        panel_print(tag="check_string_llm", content=f"Incoming --> {x}.\n{resp}", border_style="yellow")
        if "STRINGS_SAME" in resp: return True                                         
        if "STRINGS_DIFF" in resp: return False   
        
# ============================================================================:
def __handle_cli_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--backend", type=str, default="hf-local")
    parser.add_argument("--domain", type=str, default="biomedical")
    parser.add_argument("--num-ques", type=int, default=1)
    parser.add_argument("--embedder", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--qid", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    args = parser.parse_args()
    return args
# ============================================================================:
if __name__ == "__main__": 
    args     = __handle_cli_args() 
    model    = args.model 
    backend  = args.backend
    domain   = args.domain   
    num_ques = args.num_ques
    embedder = args.embedder

    if args.qid is not None: 
        do_qid = int(args.qid)
        start_idx = int(do_qid)
    else: 
        start_idx = int(args.start_idx)
        do_qid = None 

    if domain.startswith("maple"): 
        _domain = "maple"
    else: 
        _domain = domain
    config = {
        "model"   : model, 
        "backend" : backend , 
        "domain"  : domain,  # This is passed to Retriever class 
        "system_prompt" : SYSTEM_PROMPT, 
        "examples" : EXAMPLES[_domain], 
        "prefix_prompt": PREFIX_PROMPT, 
        "graph_definition": GRAPH_DEF[_domain], 
        "gen_params" : {"temperature":0.2, "max_new_tokens":256, "stop":["Observation"],}, 
        "graph_path" : graph_paths[domain], 
        "embedder_model" : embedder,  
    }
    ga = GraphAgent(**config)
    # ........................................................................:
    qa_data = load_jsonl(qa_paths[domain])                                      
    ques_answered = 0 
    print(f"DEBUG: start-idx={start_idx}")
    print(f"DEBUG: do_qid={do_qid}")
    for i, qdata in enumerate(qa_data):                                         
        # If users provided QID, then only do that one
        if do_qid and (do_qid != i): continue 

        print("X")
        # If start-idx less than i, skip 
        if i < start_idx : continue 
        print("Y")
        
        qid      = qdata['qid']
        answer   = qdata['answer']                                              
        question = qdata['question']                                            

        panel_print(tag=f"Domain-{domain}: Question ({qid}):",content=question)
        panel_print(tag=f"Domain-{domain}:   Answer ({qid}):",content=answer)
        proceed() 

        if ques_answered >= 1: 
            print(f"\nClearing memory. Setting 5 seconds timeout...")
            torch.mps.empty_cache()  # Clear cached memory
            gc.collect()  # Trigger garbage collection
            time.sleep(5)
  
     
        ga.reset()                   # TODO: why here? This should be better integrated
        for step in range(ga.steps): # TODO: This should probably be consumed by evaulate...
            raw_resp, filtered_resp = ga.evaluate(question, ground_truth=answer)
            panel_print(tag=f"GraphAgent -- Raw Response -- {step}", content=raw_resp)
            panel_print(tag=f"GraphAgent -- Filtered Response -- {step}", content=filtered_resp)
            if ga.is_finished(): break 
            proceed()

        if ga.is_finished(): 
            panel_print(tag="Final Scratchpad", content="\n".join(ga.scratchpad))
            panel_print(tag="Ground-truth", content=answer) 
            panel_print(tag="Correct?", content=ga._correct)
            print(f"Question: {question}")
            print("\nthought - trace")
            print(  "---------------")
            print("\n".join(ga.scratchpad))
            print(f"\nGround Truth: {answer}")
            # TODO: Store results 
        elif (step == ga.steps - 1): # read out on the last step
            print(f"Question: {question}")
            print("\nthought - trace")
            print(  "---------------")
            print("\n".join(ga.scratchpad))
            print(f"\nGround Truth: {answer}")
           
        ques_answered += 1
        proceed()
   
        print(f"DEBUG: ques_answered = {ques_answered} && num_ques = {num_ques}")
        if ques_answered == num_ques: break 

        
