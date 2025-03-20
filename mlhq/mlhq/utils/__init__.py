import sys 
import json 
from datetime import datetime
# --------------------------------------------------------------------|-------:
def proceed(skip=False):                                                                     
    if skip: return 
    _proceed = input("\nContinue (y/n)? ")                                         
    if _proceed in ["Y", 'y']: return                                              
    print("Exiting")                                                               
    sys.exit(0)      
# --------------------------------------------------------------------|-------:
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data
# --------------------------------------------------------------------|-------:
def load_jsonl(jsonl):                                                                                                  
    data = []                                                                   
    with open(jsonl, 'r') as file:                                              
        for line in file:                                                       
            if line.strip():                                                    
                data.append(json.loads(line))                                   
    return data 
# --------------------------------------------------------------------|-------:
def write_json(data, filename, indent=4): 
    with open(filename, 'w') as f:
        json.dump(data, f, indent=indent)
# --------------------------------------------------------------------|-------:
def get_datetime_str(): 
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# --------------------------------------------------------------------|-------:
