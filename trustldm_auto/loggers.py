import logging 
import json
import os
from datetime import datetime


def setup_logger():
    logger = logging.getLogger('PAIR')
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    return logger

def set_logger_level(logger, verbosity):
    if verbosity == 0:
        level=logging.CRITICAL # Disables logging
    elif verbosity == 1:
        level = level=logging.INFO
    else:
        level = logging.DEBUG
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
    

logger = setup_logger()
logger.set_level = lambda verbosity : set_logger_level(logger, verbosity)


class BudgetTracker:
    """Track attacker calls and judge API usage for one sample."""

    def __init__(self, attack_model: str, judge_model: str):
        self.attack_model = attack_model
        self.judge_model = judge_model
        self.attack_query_count = 0
        self.judge_query_count = 0
        self.judge_api_query_count = 0
        self.judge_input_tokens = 0
        self.judge_output_tokens = 0
        self.judge_total_tokens = 0
        self.judge_api_models = set()
        self.stage_breakdown = {}

    def _stage(self, name):
        name = name or "unknown"
        return self.stage_breakdown.setdefault(name, {
            "judge_query_count": 0,
            "judge_api_query_count": 0,
            "judge_input_tokens": 0,
            "judge_output_tokens": 0,
            "judge_total_tokens": 0,
        })

    def record_attack_queries(self, count):
        self.attack_query_count += int(count)

    def record_judge_queries(self, count, stage="unknown", api=False, model_name=None):
        count = int(count)
        bucket = self._stage(stage)
        self.judge_query_count += count
        bucket["judge_query_count"] += count
        if api:
            self.judge_api_query_count += count
            bucket["judge_api_query_count"] += count
        if model_name:
            self.judge_api_models.add(model_name)

    def record_judge_tokens(self, input_tokens=0, output_tokens=0,
                            total_tokens=None, stage="unknown", model_name=None):
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        total_tokens = int(total_tokens if total_tokens is not None
                           else input_tokens + output_tokens)
        bucket = self._stage(stage)
        self.judge_input_tokens += input_tokens
        self.judge_output_tokens += output_tokens
        self.judge_total_tokens += total_tokens
        bucket["judge_input_tokens"] += input_tokens
        bucket["judge_output_tokens"] += output_tokens
        bucket["judge_total_tokens"] += total_tokens
        if model_name:
            self.judge_api_models.add(model_name)

    def to_dict(self):
        return {
            "attack_model": self.attack_model,
            "judge_model": self.judge_model,
            "attack_query_count": self.attack_query_count,
            "judge_query_count": self.judge_query_count,
            "judge_api_query_count": self.judge_api_query_count,
            "judge_input_tokens": self.judge_input_tokens,
            "judge_output_tokens": self.judge_output_tokens,
            "judge_total_tokens": self.judge_total_tokens,
            "judge_api_models": sorted(self.judge_api_models),
            "stage_breakdown": self.stage_breakdown,
        }


class LocalResultLogger:
    """Local logger for recording experiment results."""
    
    def __init__(self, args, system_prompts, result_dir, budget_tracker=None):
        # Create results directory if it does not exist
        os.makedirs(f"{result_dir}/{args.attack_model}_{args.target_model}", exist_ok=True)
        
        # Create result file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result_dir}/{args.attack_model}_{args.target_model}/{timestamp}.json"
        self.filename = filename
        self.budget_tracker = budget_tracker
        
        # Store experiment configuration
        self.config = {
            "attack_model": args.attack_model,
            "target_model": args.target_model,
            "judge_model": args.judge_model,
            "keep_last_n": args.keep_last_n,
            "system_prompt": system_prompts,
            "index": args.index,
            "category": args.category,
            "goal": args.goal,
            "n_iter": args.n_iterations,
            "target_str": args.target_str,
            "n_streams": args.n_streams,
            "max_samples": getattr(args, "max_samples", 0),
            "privacy_data": getattr(args, "privacy_data", ""),
            "search_strategy": getattr(args, "search_strategy", "hs_ss"),
            "screening_topk": getattr(args, "screening_topk", None),
        }
        
        # Initialize storage for experiment data
        self.data = {
            "config": self.config,
            "results": [],
            "summary": {}
        }
        
        logger.info(f"Results will be saved to: {self.filename}")

    def _get_best_logged_result(self):
        best_result = None
        for iteration_data in self.data.get("results", []):
            for result in iteration_data.get("attack_data", []):
                score = result.get("score")
                if score is None:
                    score = result.get("judge_score")
                if score is None:
                    continue
                if best_result is None or score > best_result["score"]:
                    best_result = {
                        "iteration": result.get("iteration", iteration_data.get("iteration")),
                        "pre_context": result.get("pre_context"),
                        "post_context": result.get("post_context"),
                        "response": result.get("response", result.get("target_response")),
                        "score": score,
                    }
        return best_result

    def log_iteration(self, iteration: int, attack_list: list, response_list: list, judge_scores: list):
        # Record data for the current iteration
        iteration_data = {
            "iteration": iteration,
            "attack_data": []
        }
        
        for i, (attack, response, score) in enumerate(zip(attack_list, response_list, judge_scores)):
            item = {
                "stream": i + 1,
                "pre_context": attack["pre_context"],
                "post_context": attack["post_context"],
                "improvement": attack["improvement"],
                "target_response": response,
                "judge_score": score
            }
            iteration_data["attack_data"].append(item)
        
        self.data["results"].append(iteration_data)
        
        # Persist results to disk
        self._save_to_file()
    
    def log_with_decoding(self, iteration: int, pre_context: str, post_context: str, 
                         improvement: str, response: str, score: int, decoding: str):
        # Log result with decoding method metadata
        result = {
            "iteration": iteration,
            "pre_context": pre_context,
            "post_context": post_context,
            "improvement": improvement,
            "response": response,
            "score": score,
            "decoding": decoding
        }
        
        # Check if this iteration already exists; if not, add a new entry
        iteration_exists = False
        for item in self.data["results"]:
            if item["iteration"] == iteration:
                item["attack_data"].append(result)
                iteration_exists = True
                break
        
        if not iteration_exists:
            self.data["results"].append({
                "iteration": iteration,
                "attack_data": [result]
            })
        
        # Save to file
        self._save_to_file()
    

    def log_with_decoding_and_gen_length(self, iteration: int, pre_context: str, post_context: str, 
                         improvement: str, response: str, score: int, decoding: str, gen_length: int):
        # Log result with decoding method and generated length metadata
        result = {
            "iteration": iteration,
            "pre_context": pre_context,
            "post_context": post_context,
            "improvement": improvement,
            "response": response,
            "score": score,
            "decoding": decoding,
            "gen_length": gen_length
        }
        
        # Check if this iteration already exists; if not, add a new entry
        iteration_exists = False
        for item in self.data["results"]:
            if item["iteration"] == iteration:
                item["attack_data"].append(result)
                iteration_exists = True
                break
        
        if not iteration_exists:
            self.data["results"].append({
                "iteration": iteration,
                "attack_data": [result]
            })
        
        # Persist results to disk
        self._save_to_file()

    def finish(self, is_jailbroken=False, query_to_jailbreak=None, jailbreak_prompt=None, jailbreak_response=None):
        fallback_best_result = None
        if jailbreak_prompt is None or jailbreak_response is None:
            fallback_best_result = self._get_best_logged_result()
            if fallback_best_result is not None:
                if jailbreak_prompt is None:
                    jailbreak_prompt = (
                        f'{fallback_best_result["pre_context"]}<split>{fallback_best_result["post_context"]}'
                    )
                if jailbreak_response is None:
                    jailbreak_response = fallback_best_result["response"]

        # Save final summary information
        self.data["summary"] = {
            "is_jailbroken": is_jailbroken,
            "query_to_jailbreak": query_to_jailbreak,
            "jailbreak_prompt": jailbreak_prompt,
            "jailbreak_response": jailbreak_response,
            "total_iterations": len(self.data["results"])
        }
        if fallback_best_result is not None:
            self.data["summary"]["best_score"] = fallback_best_result["score"]
        if self.budget_tracker is not None:
            self.data["summary"]["budget"] = self.budget_tracker.to_dict()
        
        # Save results file
        self._save_to_file()
        logger.info(f"Experiment results saved to: {self.filename}")
    
    def _save_to_file(self):
        print("Experiment results saved to:", self.filename)
        print()
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
