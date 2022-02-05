from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Type, Any
from omegaconf import II, MISSING
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import hydra

ROOT = Path(__file__).parent
run_data_root = ROOT / 'run_data'


@dataclass
class DATASET:
    name: str
    IN_CHANNEL: int
    N_CLASSES: int
    IMG_SIZE: Tuple[int]
    data_root: str = II('data_root')


@dataclass
class FashionMNIST(DATASET):
    _target_: str = "dataset_loaders.load_fashion_mnist"
    name: str = "FashionMNIST"
    IN_CHANNEL: int = 1
    N_CLASSES: int = 10
    IMG_SIZE: Tuple[int] = field(default_factory=lambda: (28, 28))

@dataclass
class MNIST(DATASET):
    _target_: str = "dataset_loaders.load_mnist"
    name: str = "MNIST"
    IN_CHANNEL: int = 1
    N_CLASSES: int = 10
    IMG_SIZE: Tuple[int] = field(default_factory=lambda: (28, 28))


@dataclass
class CIFAR10(DATASET):
    _target_: str = "dataset_loaders.load_CIFAR10"
    name: str = "CIFAR10"
    IN_CHANNEL: int = 3
    N_CLASSES: int = 10
    IMG_SIZE: Tuple[int] = field(default_factory=lambda: (32, 32))

@dataclass
class CIFAR100(DATASET):
    _target_: str = "dataset_loaders.load_CIFAR100"
    name: str = "CIFAR100"
    IN_CHANNEL: int = 3
    N_CLASSES: int = 100
    IMG_SIZE: Tuple[int] = field(default_factory=lambda: (32, 32))


@dataclass
class Output:
    _target_: str  = "dynamics.output_coordinates.DefaultOutputFun"

@dataclass
class FirstNOutput:
    _target_: str = "dynamics.output_coordinates.FirstNOutput"
    out_size:int = II('dataset.N_CLASSES')

@dataclass
class DefaultInitFun:
    _target_: str = "dynamics.init_coordinates.DefaultInitFun"
    h_dims: Tuple[int] = field(default_factory=lambda: (II("dataset.N_CLASSES"),))
    param_map: Optional[Any] = MISSING

@dataclass
class SimpleFeatures:
    _target_: str = "models.SimpleFeatures"
    last_chan:int = 64
    activation: str = 'PReLU'
    n_in_channels:int = II('dataset.IN_CHANNEL')
    bottleneck: str = 'conv'
    conv_bias: bool = True

@dataclass
class SimpleFeatureTensor:
    _target_: str = "models.SimpleFeatureTensor"
    last_chan:int = 64
    activation: str = 'PReLU'
    n_in_channels:int = II('dataset.IN_CHANNEL')
    conv_bias: bool = True

@dataclass
class ClassDyn:
    _target_: str  = "dynamics.classification.ClassDyn"
    n_hidden: int = II('..init_fun.h_dims[0]')
    activation: str = 'PReLU'
    dropout:float = 0.5
    mlp_size:int = 128
    n_param_features:int = II('..init_fun.param_map.n_outputs')
    gain:float = 50.
    restrict_to_simplex:bool=True

@dataclass
class VaryingClassDyn:
    _target_: str  = "dynamics.classification.VaryingClassDyn"
    n_in_channels: int = II('..init_fun.param_map.last_chan')
    n_hidden: int = II('..init_fun.h_dims[0]')
    activation: str = 'PReLU'
    dropout:float = 0.5
    mlp_size:int = 128
    last_chan:int = 64


@dataclass
class ClassicalModel:
    # name:str = MISSING
    n_in_channels:int = II('dataset.IN_CHANNEL')
    n_outputs:int = II('dataset.N_CLASSES')

@dataclass
class ContinuousNet(ClassicalModel):
    _target_: str = 'libs.ContinuousNet.continuous_net.continuous_net.ContinuousNet'
    ALPHA:int = 16
    scheme:str = 'rk4'
    time_d:int = 3
    time_epsilon:float = II('.time_d')
    use_batch_norms:bool = True
    n_time_steps_per:int = 1
    use_skip_init:bool = False
    use_adjoint:bool = False
    activation_before_conv:bool = False

@dataclass
class AlexNet(ClassicalModel):
    # name="AlexNet"
    _target_: str  = "models.make_alex_net"

@dataclass
class VGG16(ClassicalModel):
    # name = "VGG16"
    _target_: str  = "models.make_vgg16"

@dataclass
class RESNET50(ClassicalModel):
    # name = "RESNET50"
    _target_: str  = "models.make_resnet50"

@dataclass
class RESNET18(ClassicalModel):
    # name = "RESNET18"
    _target_: str  = "models.make_resnet18"

@dataclass
class RESNET18Features():
    _target_: str = "models.resnet18_features"
    n_in_channels:int = II('dataset.IN_CHANNEL')
    last_chan:int = 512

@dataclass
class GeneralModule:
    decay_epochs: List[int] = field(default_factory=lambda: [30, 60, 90])
    weight_decay: float = 0.0
    lr: float = 1e-3
    opt_name: str = 'SGD'
    momentum: float = 0.9
    beta1:float = 0.9
    beta2:float = 0.999
    eps:float = 1e-8


@dataclass
class ClassicalModule(GeneralModule):
    _target_: str  = "pl_modules.ClassicalLearning"
    model:ClassicalModel = MISSING


@dataclass
class ODEModule(GeneralModule):
    _target_: str  = "pl_modules.ODELearning"
    dynamics:Any = MISSING
    output:Any = MISSING
    init_fun:Any = MISSING
    n_input: int = II('dataset.IN_CHANNEL')
    n_output: int = II('dataset.N_CLASSES')
    t_max:float = 1.0
    train_ode_solver: str = 'dopri5'
    train_ode_tol: float = 1e-6
    val_ode_solver: str = 'dopri5'
    val_ode_tol: float = 1e-6


@dataclass
class Lyapunov(ODEModule):
    _target_: str  = "pl_modules.LyapunovLearning"
    order: int = 1
    h_sample_size: int = 128
    h_dist_lim:float = 30



@dataclass
class PILyuapunov(Lyapunov):
    _target_: str  = "pl_modules.PILyapunovLearning"
    t_upper: float = 1.0
    t_delta: float = 1e-2
    minimum_effective_batch_size: int = 10

@dataclass
class ContinuousNetLyapunov(GeneralModule):
    _target_: str = "pl_modules.ContinuousNetLyapunovLearning"
    model:ContinuousNet = MISSING
    order:int =1
# ExpCfgDefaults = [
#     {"dataset": MISSING},
#     {"model":MISSING}
# ]

@dataclass
class ExpCfg:
    # defaults:List[Any] = field(default_factory=lambda: ExpCfgDefaults)
    dataset: DATASET = MISSING
    savedir: str = run_data_root
    data_root: str = ROOT / 'data'
    batch_size: int = 32
    val_batch_size: int = 32
    data_loader_workers: int = 4
    prefetch_factor: int = 4
    disable_logs: bool = False
    module: GeneralModule = MISSING
    max_epochs:int=120
    gpus:int = 0
    seed: int = 0

@dataclass
class RobustExpCfg(ExpCfg):
    model_file: str = MISSING
    norm:str = "2" # only 2 or inf

cs = ConfigStore.instance()


# cs.store(group='dataset', name='ImageNet', node=ImageNet)
cs.store(group='dataset', name='MNIST', node=MNIST)
cs.store(group='dataset', name='FashionMNIST', node=FashionMNIST)
cs.store(group='dataset', name='CIFAR10', node=CIFAR10)
cs.store(group='dataset', name='CIFAR100', node=CIFAR100)
cs.store(group='module/init_fun/param_map', name="RESNET18", node=RESNET18)
cs.store(group='module/init_fun/param_map', name="SimpleFeatures", node=SimpleFeatures)
cs.store(group='module/init_fun/param_map', name="RESNET18Features", node=RESNET18Features)
cs.store(group='module/init_fun/param_map', name="SimpleFeatureTensor", node=SimpleFeatureTensor)
cs.store(group='module/init_fun', name="DefaultInitFun", node=DefaultInitFun)
cs.store(group='module/dynamics', name="ClassDyn", node=ClassDyn)
cs.store(group='module/dynamics', name="VaryingClassDyn", node=VaryingClassDyn)
cs.store(group='module/output', name="Output", node=Output)
cs.store(group='module/output', name="FirstNOutput", node=FirstNOutput)
cs.store(group='module/model', name="AlexNet", node=AlexNet)
cs.store(group='module/model', name="VGG16", node=VGG16)
cs.store(group='module/model', name="RESNET18", node=RESNET18)
cs.store(group='module/model', name="RESNET50", node=RESNET50)
cs.store(group='module/model', name="ContinuousNet", node=ContinuousNet)
cs.store(group='module', name="ClassicalModule", node=ClassicalModule)
cs.store(group='module', name="ODEModule", node=ODEModule)
cs.store(group='module', name="Lyapunov", node=Lyapunov)
cs.store(group='module', name="PILyuapunov", node=PILyuapunov)
cs.store(group='module', name="ContinuousNetLyapunov", node=ContinuousNetLyapunov)
cs.store(name='default', node=ExpCfg)
cs.store(name='robust', node=RobustExpCfg)