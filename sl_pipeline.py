import hydra
import ExpConfig
from omegaconf import OmegaConf

import pytorch_lightning as pl
from dataset_loaders import load_mnist, load_fashion_mnist, load_CIFAR10, \
    load_CIFAR100, load_imagenet
from dynamics.classification import ClassDyn
from dynamics.naiveResNet import ResNetOutput
from dynamics.output_coordinates import LinearLastOutput
from dynamics.resnet_block import ResNetBlockDyn
from models import *
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
import torch as th
from pathlib import Path
import torch

from pl_modules import ADAPTIVE_SOLVERS, FIXED_SOVLERS

ROOT = Path(__file__).parent
run_data_root = ROOT / 'run_data'
config_path = ROOT / 'configs' / 'classification'
tensorboard_root = run_data_root / 'tensorboards'

SOVLERS = ADAPTIVE_SOLVERS + FIXED_SOVLERS


def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


# Then in training code before the train loop
set_debug_apis(state=False)


DATASETS = {'MNIST': load_mnist,
            'FashionMNIST': load_fashion_mnist,
            'CIFAR10': load_CIFAR10,
            'CIFAR100': load_CIFAR100,
            'ImageNet': load_imagenet}
IN_CHANNEL = {
             'MNIST': 1,
            'FashionMNIST': 1,
            'CIFAR10': 3,
            'CIFAR100': 3,
            'ImageNet': 3}
N_CLASSES = {'MNIST': 10,
            'FashionMNIST': 10,
            'CIFAR10': 10,
            'CIFAR100': 100,
            'ImageNet': 1000}
IMG_SIZE = {'MNIST': (32,32),
            'FashionMNIST': (32,32),
            'CIFAR10': (32,32),
            'CIFAR100': (32,32),
            'ImageNet': (32,32)}
CLASSICAL_MODELS = {
'AlexNet': make_alex_net,
'VGG16': make_vgg16,
'ResNet18': make_resnet18,
'ResNet50' : make_resnet50
}
OPTIMS = ['Adam', "AdamW", 'SGD', 'RMSprop']
MODELS = [ "LyaNet", "NeuralOde"] + list(CLASSICAL_MODELS.keys())

DYN_MODEL = {'ClassDyn': ClassDyn,
             "ResNetBlockDyn": ResNetBlockDyn}
OUTPUT_FUNS = {
    "ResNetOut": lambda in_size, out_size: ResNetOutput(out_size, in_size),
    'LinearLastOutput': LinearLastOutput,
    'None': lambda x, y: DefaultOutputFun()
}

def get_arch_name(config):
    return f'{get_module_name(config)}({get_model_name(config)})'


def get_model_name(config):
    if hasattr(config.module, "model"):
        return str(config.module.model._metadata.object_type).split('.')[-1][
               :-2]
    else:
        init_fun_name = str(
            config.module.init_fun.param_map._metadata.object_type).split('.')[-1][:-2]
        dyn_name = str(
            config.module.dynamics._metadata.object_type).split('.')[-1][:-2]
        out_name = str(
            config.module.output._metadata.object_type).split('.')[-1][:-2]
        return f"{init_fun_name},{dyn_name},{out_name}"

def get_module_name(config):
    return str(config.module._metadata.object_type).split('.')[-1][:-2]

class TimingCallback(pl.Callback):
    def on_train_epoch_start(self, trainer: 'pl.Trainer',
                             pl_module: 'pl.LightningModule') -> None:
        self.train_start = torch.cuda.Event(enable_timing=True)
        self.train_end = torch.cuda.Event(enable_timing=True)
        self.train_start.record()

    def on_train_epoch_end(self, trainer: 'pl.Trainer',
                           pl_module: 'pl.LightningModule') -> None:
        self.train_end.record()
        th.cuda.synchronize()
        elapsed_seconds = self.train_start.elapsed_time(self.train_end) / 1000
        pl_module.log('train_epoch_time', elapsed_seconds, logger=True,
                      on_epoch=True)

    def on_validation_epoch_start(self, trainer: 'pl.Trainer',
                                  pl_module: 'pl.LightningModule') -> None:
        self.val_start = torch.cuda.Event(enable_timing=True)
        self.val_end = torch.cuda.Event(enable_timing=True)
        self.val_start.record()

    def on_validation_epoch_end(self, trainer: 'pl.Trainer',
                                pl_module: 'pl.LightningModule') -> None:
        self.val_end.record()
        th.cuda.synchronize()
        elapsed_seconds = self.val_start.elapsed_time(self.val_end) / 1000
        pl_module.log('val_epoch_time', elapsed_seconds, logger=True,
                      on_epoch=True)


class SLExperiment:

    def __init__(self, config, other_callbacks=None, progbar=True, project='LyaNet'):
        OmegaConf.resolve(config)
        self.config = config

        logger = pl_loggers.TensorBoardLogger(
            save_dir=str(tensorboard_root / self.create_log_name()), flush_secs=1)
        callbacks = [TimingCallback()]
        if not config.disable_logs:
            callbacks += [
                pl.callbacks.LearningRateMonitor(logging_interval='step'),
                pl.callbacks.ModelCheckpoint(monitor='validation_error',
                                             save_top_k=1,
                                             mode='min',),
                #Disabled to keep tuning simple for now
                pl.callbacks.StochasticWeightAveraging(
                    swa_epoch_start=(1/2),
                    swa_lrs=self.config.module.lr * 1e-1
                )
            ]
        if other_callbacks is not None:
            callbacks += other_callbacks
        self.trainer = pl.Trainer(
            logger=logger if not config.disable_logs else False,
            max_epochs=config.max_epochs,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            default_root_dir=str(run_data_root / 'lightning_logs' / self.create_log_name()),
            gpus=config.gpus,
            # checkpoint_callback=not config.disable_logs,
            callbacks=callbacks,
            progress_bar_refresh_rate=None if progbar else 0,
            # gradient_clip_val=1e3,
        )
        pl.seed_everything(config.seed)

        h_param_dset = dict(config)
        h_param_dset['module'] = dict(h_param_dset['module'])
        h_param_dset['module']['name'] = get_arch_name(self.config)
        logger.log_hyperparams(h_param_dset)

    @property
    def module_name(self):
        return get_module_name(self.config)

    @property
    def model_name(self):
        return get_model_name(self.config)


    def create_log_name(self):
        returns = ''
        # returns += f'{self.exp_prefix}'
        returns += f'd.{self.config.dataset.name}'
        returns += f'_m.{self.module_name}({self.model_name})'
        returns += f'_b.{self.config.batch_size}'
        returns += f'_lr.{self.config.module.lr}'
        returns += f'_wd.{self.config.module.weight_decay}'
        returns += f'_mepoch{self.config.max_epochs}'
        returns += f'._sd{self.config.seed}'
        # returns += f'{self.exp_postfix}'
        return returns

    def run(self, checkpoint_module=None, test_only=False):
        train_loader, val_loader, test_loader = self.make_dataloaders()
        if checkpoint_module is None:
            module = hydra.utils.instantiate(self.config.module)
        else:
            module = checkpoint_module
        if not test_only:
            self.trainer.fit(module,
                             train_loader,
                             val_loader)
        self.trainer.test(module, test_loader)
        return module

    def make_dataloaders(self):
        train, val, test = self.make_datasets()
        prefetch_factor = self.config.prefetch_factor
        train_loader = DataLoader(train, batch_size=self.config.batch_size,
                                  num_workers=self.config.data_loader_workers,
                                  shuffle=True, pin_memory=True,
                                  prefetch_factor=prefetch_factor)
        eval_batch_size = self.config.val_batch_size
        val_loader = DataLoader(val, batch_size=eval_batch_size,
                                num_workers=self.config.data_loader_workers,
                                pin_memory=True,
                                prefetch_factor=prefetch_factor)
        test_loader = DataLoader(test, batch_size=eval_batch_size,
                                 num_workers=self.config.data_loader_workers,
                                 pin_memory=True,
                                 prefetch_factor=prefetch_factor)
        return train_loader, val_loader, test_loader

    def make_datasets(self):
        return hydra.utils.instantiate(
            OmegaConf.masked_copy(self.config.dataset,
                                  ['_target_', 'data_root']))


@hydra.main(config_path=str(config_path),
            config_name='classical')
def main(cfg: ExpConfig.ExpCfg) -> None:
    OmegaConf.resolve(cfg)
    print(cfg)
    pipeline = SLExperiment(cfg)
    pipeline.run()

if __name__ == '__main__':
    main()
