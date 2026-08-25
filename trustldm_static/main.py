from typing import Union, Dict

import hydra
from importlib import import_module
from trustldm_static.configs.configs import BaseConfig
from omegaconf import OmegaConf, DictConfig
from hydra.core.config_store import ConfigStore
from trustldm_static.summarize import summarize_results
import os
os.makedirs('./.cache', exist_ok=True)


PERSPECTIVES = {
    "fairness": "trustldm_static.perspectives.fairness.fairness_evaluation",
    "privacy": "trustldm_static.perspectives.privacy.privacy_evaluation",
    "safety": "trustldm_static.perspectives.safety.safety_evaluation"
}


cs = ConfigStore.instance()
cs.store(name="config", node=BaseConfig)
cs.store(name="slurm_config", node=BaseConfig)
cs.store(name="joblib_config", node=BaseConfig)
cs.store(name="llada_eval", node=BaseConfig)
cs.store(name="dream_eval", node=BaseConfig)
cs.store(name="llada_1___5", node=BaseConfig)
cs.store(name="llada_moe", node=BaseConfig)


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(raw_config: Union[DictConfig, Dict]) -> None:
    # The 'validator' methods will be called when you run the line below
    config: BaseConfig = OmegaConf.to_object(raw_config)

    if not isinstance(config, BaseConfig):
        raise ValueError(f"Wrong type of configuration generated: {type(config)}")

    ran_perspective = []
    for name, module_name in PERSPECTIVES.items():
        if getattr(config, name) is not None:
            perspective_module = import_module(module_name)
            perspective_module.main(config)
            ran_perspective.append(name)

    # summarize_results()


if __name__ == "__main__":
    main()
