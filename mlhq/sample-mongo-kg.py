import json
import os
import sys 
from pymongo import MongoClient
import argparse

DEFAULT_GRAPH_JSON = "~/code/Graph-CoT/data/processed_data/biomedical/graph.json"
DEFAULT_DB_URI = 'mongodb://localhost:27017/'
DEFAULT_DB_NAME = "graph-cot"
DEFAULT_DB_COLLECTION = "biomedical"
# --------------------------------------------------------------------|-------:
def __handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-uri', type=str, default=DEFAULT_DB_URI)
    parser.add_argument('--db-name', type=str, default=DEFAULT_DB_NAME)
    parser.add_argument('--db-collection', type=str, default=DEFAULT_DB_COLLECTION)
    parser.add_argument('--input-json', type=str, default=DEFAULT_GRAPH_JSON)

    args = parser.parse_args()
    if args.input_json and (args.input_json.startswith("~")): 
        args.input_json = os.path.expanduser(args.input_json)
 
    return args
# --------------------------------------------------------------------|-------:
def print_client_db_names(db_client): 
    for db_name in db_client.list_database_names(): 
        print(f"database: {db_name}")

def print_collections(db): 
    collection_list = db.list_collection_names()
    print(f"Collections in '{args.db_name}' database: {collection_list}")
# --------------------------------------------------------------------|-------:
def import_domain_data(db, domain_name, domain_path):
    """Import data for a specific domain"""
    collection = db[domain_name]  # Each domain gets its own collection
    
    # Check if collection already has data
    if collection.count_documents({}) > 0:
        print(f"Collection {domain_name} already has {collection.count_documents({})} documents")
        return
    
    # Find and load the graph.json file
    #graph_file = os.path.join(domain_path, "graph.json")
    #if not os.path.exists(domain_path):
    #    print(f"No graph.json found for {domain_name}")
    #    return
        
    print(f"Loading data for {domain_name}...")
    with open(domain_path, 'r') as file:
        graph_data = json.load(file)
    
    # Handle biomedical domain specially due to its complex structure
    if domain_name == "biomedical":
        documents = []
        for node_type_key in graph_data:
            node_type = node_type_key.replace("_nodes", "")  # Extract type name
            nodes = graph_data[node_type_key]
            
            for node_id, node_data in nodes.items():
                # Create a document for each node
                document = {
                    "domain": "biomedical",
                    "node_type": node_type,
                    "node_id": node_id,
                    "features": node_data.get("features", {}),
                    "neighbors": node_data.get("neighbors", {})
                }
                documents.append(document)
                
                # Insert in batches of 1000 to avoid memory issues
                if len(documents) >= 1000:
                    collection.insert_many(documents)
                    documents = []
            
        # Insert any remaining documents
        if documents:
            collection.insert_many(documents)
    else:
        # For other domains, adjust based on their structure
        # This is a placeholder - you'll need to customize based on each domain's structure
        # collection.insert_one({"domain": domain_name, "data": graph_data})
        print("Note entering other domain right now ")
    
    # Create indexes for efficient querying
    collection.create_index("node_type")
    collection.create_index("node_id")
    print(f"Imported {collection.count_documents({})} documents for {domain_name}")

    

def main(args): 
    # Create Client
    db_client = MongoClient(args.db_uri)
    db = db_client[args.db_name] 
    #collection = db[args.db_collection]

    print_client_db_names(db_client)
    print_collections(db)

    # Load the graph.json file
    with open(args.input_json, 'r') as file:
        graph_data = json.load(file)


    total = 0 
    for i,k in enumerate(graph_data,start=1): 
        print(f"({i})  {k} --> {type(graph_data[k])} {len(graph_data[k])}")
        total += len(graph_data[k])
        
        #for j, k2 in enumerate(graph_data[k],start=1): 
        #    print(f"  - ({j}) {k2}  --> {type(graph_data[k][k2])}")
    
      
    
    domain_name = "biomedical"
    import_domain_data(db, domain_name, args.input_json)
    print(total)

    sys.exit(1)

## Insert data into MongoDB
## Option 1: Insert the entire graph as a single document
#collection.insert_one(graph_data)

## Option 2: Insert each node as a separate document (flattened approach)
## This makes querying specific nodes easier
#for node_type, nodes in graph_data.items():
#    for node_id, node_data in nodes.items():
#        document = {
#            'node_type': node_type,
#            'node_id': node_id,
#            **node_data  # Unpack the node data (features and neighbors)
#        }
#        collection.insert_one(document)
# --------------------------------------------------------------------|-------:
if __name__ == "__main__": 
    args = __handle_cli_args()
    main(args)
