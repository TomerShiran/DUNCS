from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Any
from omegaconf import OmegaConf
import re


@dataclass
class SimulationCommands:
    create_data: bool
    train_model: bool
    evaluate_mode: bool
    save_model: bool
    save_dataset: bool
    save_to_file: bool = False
    plot_results: bool = False
    save_plots: bool = False


@dataclass
class SystemModelParams:
    M: Any  # Number of targets
    T: int  # Snapshots
    snr: Any  # in dB
    field_type: str  # ['Far', 'Near']
    signal_nature: str  # ['coherent', 'non-coherent']
    array_form: str  # ['ula', 'mra-4', 'mra-5', 'mra-6', 'mra-7', 'mra-8']
    N: Optional[int] = None  # Number of Antennas
    signal_type: str = "NarrowBand"
    eta: float = 0.0
    bias: float = 0.0
    sv_noise_var: float = 0.0
    freq_values: list = field(default_factory=lambda: [0, 500])
    antenna_pattern: bool = False
    doa_range: list = field(default_factory=lambda: (-60, 60))
    min_gap: int = 10

    def __post_init__(self):
        self.M = normalize_range_param(self.M)
        self.snr = normalize_range_param(self.snr)
        self.doa_range = tuple(self.doa_range)

        # if N wasn't specified in YAML, infer from array_form
        if self.N is None:
            # Match MRA: 'mra-7' → N = 7
            m_mra = re.match(r"^mra-(\d+)$", self.array_form)
            if m_mra:
                self.N = int(m_mra.group(1))

            else:
                # No inference rule applies
                raise ValueError(
                    f"Cannot infer N from array_form='{self.array_form}'. "
                    "Please specify 'N' explicitly in your config."
                )


@dataclass
class TrainingParams:
    samples_size: int
    train_test_ratio: float = 0.1
    epochs: int = 150
    batch_size: int = 256
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    weight_decay: float = 1e-9
    step_size: int = 50
    gamma: float = 0.5
    scheduler: str = "StepLR"
    training_objective: str = "angle"
    balance_factor: float = 1.0
    true_doa_train: Optional[list] = None
    true_range_train: Optional[list] = None
    true_doa_test: Optional[list] = None
    true_range_test: Optional[list] = None


@dataclass
class EvaluationParams:
    criterion: list = field(default_factory=lambda: ["rmspe"])
    balance_factor: float = 1.0
    covariance_reconstruction: str = 'sample'
    models: Optional[Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    augmented_methods: Optional[list] = field(default_factory=list)
    subspace_methods: Optional[list] = field(default_factory=list)
    admm_iterations: Optional[list] = field(default_factory=list)


@dataclass
class ModelConfig:
    model_type: str
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ADMMCovarianceReconstructionParams:
    mu: float = 2.5e-3
    rho: float = 2


@dataclass
class SimulationConfig:
    system_model: SystemModelParams
    model: ModelConfig
    training: TrainingParams
    evaluation: EvaluationParams
    commands: SimulationCommands
    admm_params: ADMMCovarianceReconstructionParams
    scenario: Optional[Dict[str, list]] = field(default_factory=dict)


def load_simulation_config(path: str) -> SimulationConfig:
    base = OmegaConf.structured(SimulationConfig)
    yaml_cfg = OmegaConf.load(path)
    cfg = OmegaConf.merge(base, yaml_cfg)
    OmegaConf.resolve(cfg)
    return OmegaConf.to_object(cfg)  # Convert to regular nested dataclasses


def normalize_range_param(param: Union[int, float, list]) -> Union[int, float, tuple]:
    """
    Normalize a parameter that can be either a scalar or a range (list of two values).

    Args:
        param: A scalar value (int or float) or a list of two values representing a range.

    Returns:
        A scalar or a tuple representing a valid range.

    Raises:
        ValueError if the input is an invalid range.
    """
    if isinstance(param, list):
        if len(param) != 2:
            raise ValueError("Expected a list of two values to represent a range.")
        return tuple(param)
    elif isinstance(param, (int, float)):
        return param
    else:
        raise TypeError("Expected int, float, or list of two elements.")
