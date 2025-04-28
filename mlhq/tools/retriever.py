import glob
import os
import sys
import json
import pickle
from contextlib import nullcontext
from typing import Dict, List
import logging

import faiss
import numpy as np
import torch
#from torch.cuda import amp
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import sentence_transformers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from IPython import embed
# --------------------------------------------------------------------|-------:
import torch
torch.set_num_threads(1)
# https://www.reddit.com/r/comfyui/comments/1c0n5de/there_appear_to_be_1_leaked_semaphore_objects_to/
# --------------------------------------------------------------------|-------:

# Note - across the different domain-features we have different access keys 
NODE_TEXT_KEYS = {
    'maple': {
        'paper' : ['title'], 
        'author': ['name'], 
        'venue' : ['name']
    },
    'amazon': {
        'item' : ['title'], 
        'brand': ['name']
    },
    'biomedical': {# 11 keys (with '_node' stripped. graph['Anatomy_node']['name']['features'])
        'Anatomy'            : ['name'], 
        'Biological_Process' : ['name'], 
        'Cellular_Component' : ['name'], 
        'Compound'           : ['name'], 
        'Disease'            : ['name'], 
        'Gene'               : ['name'], 
        'Molecular_Function' : ['name'], 
        'Pathway'            : ['name'], 
        'Pharmacologic_Class': ['name'], 
        'Side_Effect'        : ['name'], 
        'Symptom'            : ['name']
    },
    'legal': {
        'opinion': ['plain_text'], 
        'opinion_cluster': ['syllabus'], 
        'docket': ['pacer_case_id', 'case_name'], # NOTE: ONLY ONE WITH TWO
        'court': ['full_name']
    },
    'goodreads': {
        'book'     : ['title'], 
        'author'   : ['name'], 
        'publisher': ['name'],
        'series'   : ['title']
    },
    'dblp': {
        'paper'  : ['title'], 
        'author' : ['name', 'organization'], # NOTE: Two entires 
        'venue'  : ['name']
    }
}

class Retriever:
    """
    The retriever class takes in a Graph (JSON) and uses a SentenceTransformer
    to Embed parts of the graph. 
    """

    #def __init__(self, args, graph, cache=True, cache_dir=None):
    def __init__(self, graph, domain, embedder_name, embed_cache = True, cache=True, cache_dir="embed_cache_dir"):
        self.logger = logging.getLogger(f"{__name__}.Retriever")
        #self.logger.info(f"Created QAClient-{self._id}")
        if domain.startswith("maple"): 
            self.logger.info(f"Changing domain to `maple`, from `{domain}`.")
            domain = "maple"
        
        self.domain = domain 
        self.node_text_keys = NODE_TEXT_KEYS[domain]
        self.use_gpu = False #args.faiss_gpu
        self.model_name = embedder_name
        self.model = sentence_transformers.SentenceTransformer(self.model_name)
        self.graph = graph
        self.cache = embed_cache
        self.cache_dir = cache_dir

        print(f"DEBUG: [Retriever]: Incoming embed cache directory: {self.cache_dir}")
        if not os.path.exists(self.cache_dir): 
            os.makedirs(self.cache_dir)
        self.reset()

    def reset(self):
        """
        Initializes the Retriever Class. 
        """
        # (f"DEBUG: [reset]: Starting...")
        docs, ids, meta_type = self.process_graph()
        save_model_name = self.model_name.split('/')[-1]
        self.logger.info(f"Save model name = {save_model_name}")

        if self.cache and os.path.isfile(os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl')):
            self.logger.info("Loading Embedding from file")
            embeds, self.doc_lookup, self.doc_type = pickle.load(open(os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl'), 'rb'))
            assert self.doc_lookup == ids
            assert self.doc_type == meta_type
        else:
            self.logger.info("Creating Embedding from file")
            embeds = self.multi_gpu_infer(docs)
            self.doc_lookup = ids
            self.doc_type = meta_type
            pickle.dump([embeds, ids, meta_type], open(os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl'), 'wb'))
        self.init_index_and_add(embeds)
      
        #for i, embed in enumerate(embeds): 
        #    print(f"DEBUG: [reset]: {i}.)  embed = {embed}")




    def process_graph(self, debug=False):
        print(f"DEBUG: [process_graph]: Starting...")
        docs = []
        ids = []
        meta_type = []

        for node_type_key in self.graph.keys():
            print(f"DEBUG: [process_graph]: node_type={node_type_key}")
            node_type = node_type_key.split('_nodes')[0] # Anatomony_nodes --> Anatomony
            logger.info(f'loading text for {node_type}')
            for nid in tqdm(self.graph[node_type_key]):
                # TODO: Loop NODE_TEXT_KEYS[domain]
                #for lookup,_list in NODE_TEXT_KEYS[domain].items(): 
                #    print(lookup, _list)
                for field in NODE_TEXT_KEYS[self.domain][node_type]: 
                    doc = self.graph[node_type_key][nid]['features'][field]
                    docs.append(doc)   
                #doc = self.graph[node_type_key][nid]['features']['name'] # nid --> DB00772, doc --> Malathion
                #docs.append(doc) # [_node][nid][features][name]
                ids.append(nid)
                meta_type.append(node_type)
        print(f"DEBUG: [process-graph]: Number of docs = {len(docs)}")
        print(f"DEBUG: [process-graph]: Number of ids  = {len(ids)}")
        print(f"DEBUG: [process-graph]: Number of Node types: {len(meta_type)}")
        #sys.exit(1)
        return docs, ids, meta_type

    def multi_gpu_infer(self, docs):
        print(f"DEBUG: [multi-gpu-infer]: Starting...")
        pool = self.model.start_multi_process_pool()
        embeds = self.model.encode_multi_process(docs, pool)
        return embeds

    def _initialize_faiss_index(self, dim: int):
        print(f"DEBUG: [_initialize_faiss_index]: Starting...")
        self.index = None
        cpu_index = faiss.IndexFlatIP(dim)
        self.index = cpu_index
        print(f"DEBUG: [_initialize_faiss_index]: cpu_index = {cpu_index}")

    def _move_index_to_gpu(self):
        logger.info("Moving index to GPU")
        print(f"DEBUG: [_move_index_to_gpu]: Starting...")
        ngpu = faiss.get_num_gpus()
        print(f"DEBUG: [_move_index_to_gpu]: num-gpus={ngpu}")
        gpu_resources = []
        for i in range(ngpu):
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)
        print(f"DEBUG: [_move_index_to_gpu]: Calling faiss.GpuMultipleClonerOptions")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        self.index = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)

    def init_index_and_add(self, embeds):
        print(f"DEBUG: [init_index_and_add]: Starting...")
        #logger.info("Initialize the index...")
        dim = embeds.shape[1]
        self._initialize_faiss_index(dim)
        self.index.add(embeds)

        if self.use_gpu:
            self._move_index_to_gpu()

    @classmethod
    def build_embeddings(cls, model, corpus_dataset, args):
        print(f"DEBUG: [build_embeddings]: Starting...")
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        return retriever

    @classmethod
    def from_embeddings(cls, model, args):
        print(f"DEBUG: [from_embeddings]: Starting...")
        retriever = cls(model, None, args)
        if args.process_index == 0:
            retriever.init_index_and_add()
        if args.world_size > 1:
            torch.distributed.barrier()
        return retriever

    def reset_index(self):
        if self.index:
            self.index.reset()
        self.doc_lookup = []
        self.query_lookup = []

    def search_single(self, query, topk: int = 10):
        # logger.info("Searching")
        if self.index is None:
            raise ValueError("Index is not initialized")
        
        query_embed = self.model.encode(query, show_progress_bar=False)

        D, I = self.index.search(query_embed[None,:], topk)
        original_indice = np.array(self.doc_lookup)[I].tolist()[0][0]
        original_type = np.array(self.doc_type)[I].tolist()[0][0]

        return original_indice, self.graph[f'{original_type}_nodes'][original_indice]


if __name__ == '__main__':
    graph_dir = "/Users/msbabo/code/Graph-CoT/data/processed_data/biomedical/graph.json"
    #node_text_keys = NODE_TEXT_KEYS[domain]

    graph = json.load(open(graph_dir))
    domain = 'biomedical'
    model_name = "sentence-transformers/all-mpnet-base-v2"
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- :
    node_retriever = Retriever(graph, domain, model_name)
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- :
    query = "quantum physics and machine learning"
    query = "Malathion"
    idd, node = node_retriever.search_single(query, 1)
    print(idd, node)
    print("finished")





















