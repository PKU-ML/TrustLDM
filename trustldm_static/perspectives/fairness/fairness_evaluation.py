import os
import json
import numpy as np
from tempfile import TemporaryDirectory
from fairlearn.metrics import equalized_odds_difference
from fairlearn.metrics import demographic_parity_difference
from datasets import load_dataset

from trustldm_static.perspectives.fairness.fair_chat import do_classification
from trustldm_static.perspectives.fairness.task_description import task_dic
from trustldm_static.chat import Chat
from trustldm_static.perspectives.fairness.score_calculation_script import aggregate_current_scores


def main(args):
    # load dataset
    dataset = []
    # dataset_hg = load_dataset("AI-Secure/DecodingTrust", 'fairness', data_files=[f'fairness/fairness_data/{args.fairness.prompt_file}'])
    # dataset_hg = load_dataset('jsonl', data_files=f"data/fairness/fairness_data/{args.fairness.prompt_file}")
    # dataset_hg = dataset_hg['train']
    # for row in dataset_hg:
    #     dataset.append(row)

    with open(args.fairness.data_dir + args.fairness.prompt_file) as f:
        for line in f.readlines():
            dataset.append(json.loads(line))
    
    # start evaluation

    with TemporaryDirectory(dir="./.cache") as dirname:
        if args.dry_run:
            model = Chat.from_helm(args, cache=dirname)
            acc, unknown, (cost, prompt_tokens, cont_tokens), cache = model.do_classification(dataset, task_message=task_dic[args.fairness.dataset], example_prefix=True, dry_run=args.dry_run)

            print("cost:", cost)
            print("#/ prompt tokens:", prompt_tokens)
            print("#/ completion tokens:", cont_tokens)
        else:
            Chat.do_classification = do_classification
            model = Chat.from_helm(args, cache=dirname)
            
            task_message = task_dic[args.fairness.dataset]
            acc, unknown, cost, cache, prediction = model.do_classification(dataset, task_message, example_prefix=True, args=args)

            # record results
            out = {
                "requests": [x[0] for x in cache],
                "responses": [x[1] for x in cache],
                "labels": [x["label"] for x in dataset],
                "scores": {"accuracy": acc}
            }
            os.makedirs(os.path.dirname(args.fairness.out_file), exist_ok=True)
            if 'adult_context' not in args.fairness.out_file:
                with open(args.fairness.out_file, "w") as dest:
                    json.dump(out, dest)
            else:
                mode = args.fairness.mode
                mode = "no_context" if mode is None else mode
                print(mode)
                with open(args.fairness.out_file.removesuffix(".json") + '_' + mode + ".json", "w") as dest:
                    json.dump(out, dest)

            # Load labels and sensitive attributes
            # Process labels and sensitive attributes for the new dataset
            # Aggregate scores
            aggregate_current_scores(args)
            # print("000001111111111111111")
