import os
from pathlib import Path
from shutil import rmtree
from dotmap import DotMap
import pandas as pd 
import json 

def save_json(data, filename):
    """ Saves python data as json. """
    with open(filename, 'w') as fp:
        json.dump(data, fp, indent=3)

def json2data(filename):
    """ loads a json file into a list (od dictionaries) """
    
    with open(filename,'r') as json_file:
        data = json.load(json_file)

    return data

def json2dataframe(filename):
    """loads a json file into a pandas datafframe """
    return pd.DataFrame(json2data(filename))

def read_config(filename):
    """ Read a json file and transforms into a DotMap. """
    return DotMap(json2data(filename))

def create_dir(path, clear=False):
    """ Creates a directory on disk. """

    if clear and os.path.exists(path):
        rmtree(path)
    Path(path).mkdir(parents=True, exist_ok=not clear) 

def save(file, text):
    """ Saves a text in a text file. """
    with open(file, 'w') as fp:
        fp.write(str(text))