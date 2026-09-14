from dataclasses import dataclass
from enum import Enum

class NetworkType(Enum):
    NONE = 2
    NVP = 3
    RANDOM = 4
    SPEC_FLOW = 5

class AugmentationType(Enum):
    GAUSSIAN_NOISE = 0
    MIXUP = 2
    CUTMIX = 3
    RANDOM = 4


class AugmentationMOMENT(Enum):
    PRE_TRANSFORM = 1
    POST_TRANSFORM = 2
    RANDOM = 3
    NONE = 4

class SortLossType(Enum):
    LOGISTIC = 1
    NONE = 3

class BaseLossType(Enum):
    MSE = 0
    NONE = 1

class DistributionType(Enum):
    STUDENT_T = "student_t"
    LAPLACE = "laplace"
    CAUCHY = "cauchy"

class Nonlinearity(Enum):
    PIECEWISE = "piecewise"
    ELU = "elu"
    SOFTPLUS = "softplus"

class MixtureType(Enum):
    GAUSSIAN = 0
    HEAVY_TAIL = 1
    RANDOM = 2

@dataclass
class GMMConfig:
    seed: int = 42
    output_dir: str = 'results'
    model_path: str = ""
    testsets_path: str = ""

    sort_loss_type: SortLossType = SortLossType.LOGISTIC
    base_loss_type: BaseLossType = BaseLossType.MSE
    nr_epochs: int = 2000
    learning_rate: float = 3e-5
    accum_steps: int = 25
    device: str = 'cuda'

    num_gaussians: int = 20
    cluster_min_points: int = 50
    cluster_max_points: int = 500
    n_support_min: int = 200
    n_support_max: int = 2000
    n_query: int = 256
    dim: int = 50

    network_type: NetworkType = NetworkType.RANDOM
    augment_type: AugmentationType = AugmentationType.RANDOM
    noise_std: float = 3.0
    mean_range: float = 1.0
    augment_moment: AugmentationMOMENT = AugmentationMOMENT.PRE_TRANSFORM
    mixture_type: MixtureType = MixtureType.RANDOM
    prob_cat: float = 0.3

    estimate: bool = True
    n_series: int = 2
    n_hutch: int = 75

    slope_coeff: float = 0.7
    num_layers: int = 4

    num_train_datasets: int = 1000
    num_test_datasets: int = 200

    embed_dim: int = 512
    n_head: int = 4
    hid_dim: int = 1024
    n_layers: int = 12
    dropout: float = 0.0

    anomaly_stratify: bool = True
    evaluate_downstream: bool = False
