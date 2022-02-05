from omegaconf import OmegaConf
import hydra
import hydra.utils

def hydra_conf_load_from_checkpoint(chkpt_file, cfg):
    instance_args = dict()
    cfg_mask = list()
    for k in cfg.keys():
        if OmegaConf.is_dict(cfg[k]) and '_target_' in cfg[k]:
            instance_args[k] = hydra.utils.instantiate(cfg[k])
        else:
            cfg_mask += [k]
    ModuleType = type(hydra.utils.instantiate(cfg))
    return ModuleType.load_from_checkpoint(
        chkpt_file,
        map_location=lambda storage, loc: storage,
        **OmegaConf.masked_copy(cfg, cfg_mask),
        **instance_args
    )