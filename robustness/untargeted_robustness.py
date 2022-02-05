import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from ExpConfig import ExpCfg, RobustExpCfg
from sl_pipeline import config_path
from sl_pipeline import SLExperiment
from pl_modules import AdversarialLearning
from pathlib import Path
import torch as th
import os, sys
import itertools
from utils import hydra_conf_load_from_checkpoint
import torchattacks

file_path = Path(__file__).parent
config_path = Path(os.path.relpath(file_path.parent / 'configs' /
                                 'robustness', start=file_path))


@hydra.main(config_path=config_path,
            config_name='classical')
def main(cfg: RobustExpCfg) -> None:

    exp = SLExperiment(cfg, project='robustness')
    exp_name = exp.create_log_name()
    module = hydra_conf_load_from_checkpoint(
        to_absolute_path(cfg.model_file),
        exp.config.module)

    # verify model works
    # exp.run(checkpoint_module=module, test_only=True)
    if cfg.norm == "2":
        print("[INFO] Using L2 Attack")
        atk = torchattacks.PGDL2(module, steps=10, eps=127 / 255, alpha=2 / 255)
    elif cfg.norm == "inf":
        print("[INFO] Using L Infinity Attack")
        atk = torchattacks.PGD(module, steps=10, eps=8 / 255, alpha=2 / 255)
    else:
        raise RuntimeError(f"[ERROR] Unsupported norm value: {cfg.norm}.")
    try:
        #we have to reduce integration quality because attacks need to compute
        #gradients from solutions to the image.
        module.val_ode_tol = 1e-1
        module.val_ode_solver = 'rk4'
        module.train_ode_solver = 'rk4'
        module.train_ode_tol = 1e-1
        print(f'Running experiment for {exp_name}')
    except:
        print(f'Running experiment for {exp_name}')
    adv_module = AdversarialLearning(atk, module)
    exp.config.val_batch_size = 1028
    exp.run(checkpoint_module=adv_module, test_only=True)
    exp.config.val_batch_size = 1028
    adv_module.run_adv = True
    exp.run(checkpoint_module=adv_module, test_only=True)



if __name__ == '__main__':
    main()
