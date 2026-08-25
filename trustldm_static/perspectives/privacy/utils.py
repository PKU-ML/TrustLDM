
import pickle
from tqdm import tqdm
import string
import random
import json
random.seed(0)
import os
from datasets import load_dataset

def get_local_domain(email):
    return email.split('@')

def load_pickle(filename):
    with open(filename, "rb") as pickle_handler:
        results = pickle.load(pickle_handler)
    return results

def load_json(filename):
    with open(filename, "rb") as f:
        results = json.load(f)
    return results
     
def get_local_domain(email):
    return email.split('@')


def load_csv(filename):
    results = {}
    with open(filename) as f:
        for line in f.readlines()[1:]:
            email,name = line.strip().split(',')
            results[email] = name
    return results


def load_awareness_privacy_dataset(filename, context_type):
    """
    Load Awareness Dataset With Context
    return: (dataset, gt_list)
        - dataset: list of messages, format: [{"role": "...", "content": "..."}, ...]
        - gt_list: list of gt
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    dataset = []
    gt_list = []
    
    for item in data:
        # get prompt
        prompt = item['prompt']
        
        if "System:" in prompt and "Third party user:" in prompt:
            system_content = prompt.split("System:")[1].split("Third party user:")[0].strip()
            user_content = prompt.split("Third party user:")[1].strip()
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            # pre context is fixed
            if context_type != "empty":
                messages.append({"role": "pre_context", "content": "The required information is as follow:"})

            # post context is specified
            if context_type == "benign":
                messages.append({"role": "context", "content": item["post_context_benign"]}) \
                if "user" in filename else \
                messages.append({"role": "context", "content": item["post_context_benign_user"]})
            elif context_type == "concise":
                messages.append({"role": "context", "content": item["post_context_concise"]}) \
                if "user" in filename else \
                messages.append({"role": "context", "content": item["post_context_concise_user"]})
            elif context_type == "detailed":
                messages.append({"role": "context", "content": item["post_context_detailed"]}) \
                if "user" in filename else \
                messages.append({"role": "context", "content": item["post_context_detailed_user"]})
            elif context_type != "empty":
                raise NotImplementedError("unavailable context_type!")
            dataset.append(messages)
            gt_list.append(item['info-content'])
    
    return dataset, gt_list