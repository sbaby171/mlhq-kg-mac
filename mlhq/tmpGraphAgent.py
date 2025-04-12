import mlhq
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

# ============================================================================:
class GraphAgent:
    def __init__(self, 
                 graph_path=None, 
                 core_model=None, 
                 core_backend=None,
                 core_system_prompt=None,
                 judge_model=None, 
                 judge_backend=None,
                 judge_system_prompt=None,
                 **kwargs):
        """
        Initialize a GraphAgent with model configurations.
        
        Args can be passed directly or through a config dictionary using **config.
        """
        self._core_llm = None
        self._judge_llm = None
        self._graph = GraphInterface(graph_path)
        
        # ....................................................................:
        #if mlhq.check_valid_combo(core_model, core_backend):
        #    self._core_llm = mlhq.Client(core_model, core_backend)
        self._core_llm = mlhq.Client(core_model, core_backend)
        #    
        #if mlhq.check_valid_combo(judge_model, judge_backend):
        #    self._judge_llm = mlhq.Client(judge_model, judge_backend)
        self._judge_llm = mlhq.Client(judge_model, judge_backend)
        # These function should be looking into a configuration file (or 
        # loading one) then is configurable by the user. We provide a default 
        # that users are suggested to use. It contains model support, where 
        # that model is support (backend), default priorities of backend 
        # enhancement request, and token references. 
        # 
        # Also, it needs strong logging capabilities. 
        # ....................................................................:
     
    @property
    def graph(self): 
        return self._graph

# ============================================================================:
if __name__ == "__main__": 
    config = {
        "graph_path"          : "/Users/msbabo/code/Graph-CoT/data/processed_data/biomedical/graph.json", 
        "core_model"          : "meta-llama/Llama-3.2-3B-Instruct", 
        "core_backend"        : "hf-local", 
        "core_system_prompt"  : "/Users/msbabo/code/mlhq-kg-mac/mlhq/system-prompts/core-system-prompt-1.py", 
        "judge_model"         : "meta-llama/Llama-3.2-3B-Instruct", 
        "judge_backend"       : "hf-local", 
        "judge_system_prompt" : "/Users/msbabo/code/mlhq-kg-mac/mlhq/system-prompts/judge-system-prompt-1.py"
    }
    ga = GraphAgent(**config)
    print(type(ga))
    print(f"graph length: {ga.graph.__len__()}")
    print(type(ga._core_llm))
    print(type(ga._judge_llm))
    

    
    

    
