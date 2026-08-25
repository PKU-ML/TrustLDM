import json
import os

def load_dataset_safety(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    goal = []
    target = []
    
    for item in data:
        # get needed prompt
        goal.append(item['Goal'])
        target.append(item['prompt'])
    
    return goal, target
    

def load_dataset_privacy(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    goal = []
    target = []
    gt = []
    system_prompts = []

    for item in data:
        # get needed prompt
        system = item["prompt"].split("\nThird party user:\n")[0].replace("System: ", "")
        system_prompts.append(system)
        goal.append(f"Obtain {item['name']}'s {item['privacy_information']} from database precisely")
        target.append(item['question'])
        gt.append(item["info-content"])
    
    return goal, target, gt, system_prompts

def load_dataset_fairness(filename):
    dataset = []
    with open(filename) as f:
        for line in f.readlines():
            dataset.append(json.loads(line))
    
    gt = []
    goal = []
    target = []

    for item in dataset:
        goal.append(f"Make a biased judgement on gender.")
        target.append(item['input'])
        gt.append(item['label'])
    
    return goal, target, gt