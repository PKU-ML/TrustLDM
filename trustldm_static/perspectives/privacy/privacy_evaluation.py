import json
import os

from trustldm_static.chat import Chat
from tempfile import TemporaryDirectory

from trustldm_static.perspectives.privacy.result_agg import calculate_scores

class MessageConstructor(object):
    def __init__(self, sys_demo, task_desc,is_raw_input=False):
        self.sys_demo = sys_demo
        self.task_desc = task_desc
        self.is_raw_input=is_raw_input
    def get_message(self, input):
        messages = []
        if len(self.sys_demo) > 0:
            messages.append({"role": "system", "content": self.sys_demo})

        if len(self.task_desc) > 0:
            messages.append({"role": "user", "content": self.task_desc})
        if self.is_raw_input:
            messages.append({"role": "user", "content": input})   
        else:    
            messages = messages + input
            
        return messages

ScenarioName2ResultDir = {
    "awn_privacy": "./results/privacy/generations/awareness"
}

def generate_and_save(args, dataset, gt_list, constructor, out_file, batch_size, keynames=['gt'], ner=False):
    dataset_size= len(dataset)
    num_batches = int(dataset_size / batch_size)
    price_cost = 0
    prompt_tokens = 0
    comple_tokens = 0
    num_prompt = 0
    if ner==True: 
        try:
            from transformers import pipeline
            ner_classifier = pipeline("token-classification", model="/home1/ybshi/starpii", aggregation_strategy="simple")
        except Exception as err:
            print(err)

    with TemporaryDirectory(dir="./.cache") as dirname:
        model = Chat.from_helm(args, cache=dirname)

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_data = dataset[batch_start:batch_end]
            # batch_gt = gt_list[batch_start:batch_end]

            cost, cache = model.do_generation(batch_data, constructor, n=args.privacy.n, t=args.privacy.t,
                                                max_tokens=args.privacy.max_tokens,
                                                dry_run=args.dry_run)  # only works for n=1 now...

            # computing resource
            price_cost += cost[0]
            prompt_tokens += cost[1]
            comple_tokens += cost[2]
            num_prompt += len(cache)


            with open(out_file, "a") as f:
                for idx, x in enumerate(cache):
      
                    res={}
                    for i, key in enumerate(keynames):
                      
                        batch_gt = gt_list[i][batch_start:batch_end] 
                        res[key] = batch_gt[idx]

                    res_gen = { "continuation": x[1], "isbanned": x[2], "prompt": x[3], "response": x[-1]}
                    res.update(res_gen)
                    if ner==True: 
                        try:
                            ner_results = ner_classifier(x[1])
                            ner_words=[] 
                            if len(ner_results)>0:
                                ner_words= [entity['word'] for entity in ner_results]
                            res.update({"named_entity_rec_continuation":  ner_words })
                        except Exception as err:
                            print(err)
                    f.write(json.dumps(res) + "\n")
    print("Cost: ", price_cost)
    print("Num of prompt tokens: ", prompt_tokens)
    print("Num of completion tokens: ", comple_tokens)
    return price_cost, prompt_tokens, comple_tokens, num_prompt

def main(args):
    try:
        args.model_config.model += ( "_"+args.model_config.remasking)
    except Exception as e:
        print("No remasking, using default remasking")
    try:
        args.model_config.model += ("_steps" + str(args.model_config.steps))
    except Exception as e:
        print("No steps, using default steps")
    try:
        args.model_config.model += ("_len" + str(args.model_config.gen_length))
    except Exception as e:
        print("No gen_length, using default gen_length")
    try:
        args.model_config.model += ("_block_len" + str(args.model_config.block_length))
    except Exception as e:
        print("No block_length, using defalut block length")
    try:
        args.model_config.model += ("_" + args.model_config.add_role )
    except Exception as e:
        print("Do not add role")
    try:
        args.model_config.model += ("_" + str(args.privacy.seed))
    except Exception as e:
        raise ValueError("no seed is PROHIBITED!\n", e)

    print(args)
    out_file = args.privacy.out_file
    dataset_size = args.privacy.dataset_size

    all_cost = {}
    all_prompt_tokens = {}
    all_comple_tokens = {}
    all_num_prompt = {}

    if args.privacy.scenario_name == "awn_privacy":
        
        constructor = MessageConstructor("", "", is_raw_input=False)
        for context_type in args.privacy.context_type:
            print("context type:", context_type)
            # load dataset
            from trustldm_static.perspectives.privacy.utils import load_awareness_privacy_dataset
            dataset, gt_list = load_awareness_privacy_dataset(args.privacy.data_file, context_type)
            
            # create output directory
            result_dir = ScenarioName2ResultDir[args.privacy.scenario_name]
            os.makedirs(result_dir, exist_ok=True)
            
            out_file = os.path.join(
                result_dir,
                "awareness_privacy_{}_{}.jsonl".format(
                    context_type, args.model_config.model
                ).replace("/", "_").replace(" ", "")
            )
            
            print(f"Saving to {out_file}")
            
            # generations
            price_cost, prompt_tokens, comple_tokens, num_prompt = generate_and_save(
                args=args,
                dataset=dataset,
                gt_list=[gt_list],
                constructor=constructor,
                out_file=out_file,
                batch_size=args.privacy.batch_size,
                keynames=['gt']
            )
            
            all_cost["awareness_privacy"] = price_cost
            all_prompt_tokens["awareness_privacy"] = prompt_tokens
            all_comple_tokens["awareness_privacy"] = comple_tokens
            all_num_prompt["awareness_privacy"] = num_prompt
    print("*********final results summary******")

    cost_over_all_settings = 0
    prompt_tokens_over_all_settings = 0
    comple_tokens_over_all_settings = 0
    num_prompt_over_all_settings = 0
    for key in all_cost.keys():
        print("scenario: ", key)
        print("Cost: ", all_cost[key])
        print("Num of prompt tokens: ", all_prompt_tokens[key])
        print("Num of completion tokens: ", all_comple_tokens[key])
        cost_over_all_settings += all_cost[key]
        prompt_tokens_over_all_settings += all_prompt_tokens[key]
        comple_tokens_over_all_settings += all_comple_tokens[key]
        num_prompt_over_all_settings += all_num_prompt[key]

    print("*********sum results summary******")
    print("sum - Cost: ", cost_over_all_settings)
    print("sum - Num of prompt tokens: ", prompt_tokens_over_all_settings)
    print("sum - Num of completion tokens: ", comple_tokens_over_all_settings)
    print("sum - Num of prompt: ", num_prompt_over_all_settings)
    
    print("*********calculate score ******")
    calculate_scores()