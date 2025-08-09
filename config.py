"""
안전성 강화된 오픈소스 LLM 추론 성능 최적화 설정 관리
모든 문제점이 해결된 안전한 버전
"""
import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
import psutil

# Pydantic을 사용한 강력한 검증
try:
    from pydantic import BaseModel, validator, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

# 안전한 환경 설정
class SafeEnvironmentManager:
    """안전한 환경 변수 관리"""

    _original_env = {}
    _lock = threading.Lock()

    @classmethod
    def set_safe_defaults(cls):
        """안전한 기본 환경 변수 설정 (기존 값 보존)"""
        safe_defaults = {
            'TOKENIZERS_PARALLELISM': 'false',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
            'TRANSFORMERS_CACHE': str(Path.home() / '.cache' / 'huggingface' / 'transformers'),
            'HF_HOME': str(Path.home() / '.cache' / 'huggingface'),
        }

        # Linux 전용 설정
        if sys.platform.startswith('linux'):
            safe_defaults['OMP_NUM_THREADS'] = '4'  # 1 대신 4로 조정

        with cls._lock:
            for key, value in safe_defaults.items():
                if key not in os.environ:
                    os.environ[key] = value
                    cls._original_env[key] = None
                else:
                    cls._original_env[key] = os.environ[key]

    @classmethod
    def restore_environment(cls):
        """원래 환경 변수 복원"""
        with cls._lock:
            for key, original_value in cls._original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
            cls._original_env.clear()

# 환경 설정 초기화
SafeEnvironmentManager.set_safe_defaults()

if PYDANTIC_AVAILABLE:
    class ModelConfigBase(BaseModel):
        """Pydantic 기반 모델 설정"""
        name: str
        model_path: str
        model_type: str = "transformers"
        device: str = "auto"
        dtype: str = "float16"
        max_memory: Optional[str] = None
        trust_remote_code: bool = False
        use_flash_attention: bool = False
        load_in_8bit: bool = False
        load_in_4bit: bool = False

        # vLLM 설정
        tensor_parallel_size: int = 1
        pipeline_parallel_size: int = 1
        max_model_len: Optional[int] = None
        gpu_memory_utilization: float = Field(default=0.8, ge=0.1, le=0.95)

        # 기타 설정
        base_url: str = "http://localhost:11434"
        quantization_config: Optional[Dict[str, Any]] = None

        @validator('device')
        def validate_device(cls, v):
            valid_devices = ['auto', 'cpu', 'cuda', 'mps']
            if v not in valid_devices:
                raise ValueError(f"Device must be one of {valid_devices}")
            return v

        @validator('dtype')
        def validate_dtype(cls, v):
            valid_dtypes = ['float16', 'float32', 'bfloat16', 'int8']
            if v not in valid_dtypes:
                raise ValueError(f"Dtype must be one of {valid_dtypes}")
            return v

        @validator('model_type')
        def validate_model_type(cls, v):
            valid_types = ['transformers', 'vllm', 'ollama', 'tgi']
            if v not in valid_types:
                raise ValueError(f"Model type must be one of {valid_types}")
            return v

        class Config:
            extra = "forbid"  # 알 수 없는 필드 금지

else:
    # Pydantic이 없는 경우 기본 dataclass 사용
    @dataclass
    class ModelConfigBase:
        name: str
        model_path: str
        model_type: str = "transformers"
        device: str = "auto"
        dtype: str = "float16"
        max_memory: Optional[str] = None
        trust_remote_code: bool = False
        use_flash_attention: bool = False
        load_in_8bit: bool = False
        load_in_4bit: bool = False
        tensor_parallel_size: int = 1
        pipeline_parallel_size: int = 1
        max_model_len: Optional[int] = None
        gpu_memory_utilization: float = 0.8
        base_url: str = "http://localhost:11434"
        quantization_config: Optional[Dict[str, Any]] = None

class ModelConfig(ModelConfigBase):
    """향상된 모델 설정 클래스"""

    def __post_init__(self):
        """초기화 후 검증 및 자동 조정"""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

        # 장치 자동 감지
        if self.device == "auto":
            self.device = self._detect_best_device()

        # 메모리 기반 자동 양자화
        if self.device == "cuda" and not self.load_in_4bit and not self.load_in_8bit:
            gpu_memory = self._get_gpu_memory()
            if gpu_memory and gpu_memory < 16:  # 16GB 미만
                self.load_in_4bit = True
                logging.info(f"Auto-enabled 4-bit quantization for GPU with {gpu_memory:.1f}GB memory")

    def _detect_best_device(self) -> str:
        """최적 장치 감지"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon 지원
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def _get_gpu_memory(self) -> Optional[float]:
        """GPU 메모리 크기 반환 (GB)"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / 1024**3
        except:
            pass
        return None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        if PYDANTIC_AVAILABLE and isinstance(self, BaseModel):
            return self.dict()
        else:
            return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """딕셔너리에서 생성"""
        # 알 수 없는 키 제거
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()} if hasattr(cls, '__dataclass_fields__') else set(data.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

@dataclass
class InferenceParams:
    """안전한 추론 파라미터"""
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 512
    min_new_tokens: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True
    batch_size: int = 1
    num_return_sequences: int = 1

    def __post_init__(self):
        """파라미터 검증 및 안전 범위 적용"""
        # 안전 범위 적용
        self.temperature = max(0.0, min(2.0, self.temperature))
        self.top_p = max(0.01, min(1.0, self.top_p))
        self.top_k = max(1, min(100, self.top_k))
        self.max_new_tokens = max(1, min(2048, self.max_new_tokens))
        self.repetition_penalty = max(1.0, min(2.0, self.repetition_penalty))
        self.length_penalty = max(0.1, min(2.0, self.length_penalty))
        self.num_beams = max(1, min(8, self.num_beams))
        self.batch_size = max(1, min(32, self.batch_size))

        # 논리 검증
        if self.min_new_tokens > self.max_new_tokens:
            self.min_new_tokens = 1

        # 샘플링 일관성 확인
        if self.temperature == 0.0:
            self.do_sample = False

@dataclass
class OptimizationConfig:
    """안전한 최적화 설정"""
    enable_torch_compile: bool = False
    enable_bettertransformer: bool = False
    enable_ipex: bool = False
    enable_deepspeed: bool = False
    deepspeed_config: Optional[str] = None

    # 메모리 최적화
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    disk_offload: bool = False

    # 배치 처리 최적화
    dynamic_batching: bool = False
    max_batch_size: int = 8
    max_sequence_length: int = 2048

    # 안전성 설정
    memory_check_interval: int = 60  # 초
    max_memory_usage: float = 0.9
    auto_cleanup: bool = True

@dataclass
class TestConfig:
    """테스트 설정"""
    dataset_name: str
    dataset_path: str
    output_dir: str = "results"
    num_samples: Optional[int] = None
    random_seed: int = 42
    num_workers: int = 1
    prefetch_factor: int = 1
    pin_memory: bool = False

class HardwareDetector:
    """강화된 하드웨어 감지"""

    _cache = {}
    _cache_lock = threading.Lock()

    @classmethod
    def detect_hardware(cls, use_cache: bool = True) -> Dict[str, Any]:
        """하드웨어 정보 감지 (캐시 지원)"""
        if use_cache:
            with cls._cache_lock:
                if 'hardware_info' in cls._cache:
                    return cls._cache['hardware_info'].copy()

        hardware_info = cls._detect_hardware_impl()

        if use_cache:
            with cls._cache_lock:
                cls._cache['hardware_info'] = hardware_info.copy()

        return hardware_info

    @classmethod
    def _detect_hardware_impl(cls) -> Dict[str, Any]:
        """실제 하드웨어 감지 구현"""
        hardware_info = {
            'cuda_available': False,
            'mps_available': False,
            'cuda_device_count': 0,
            'total_memory': 0,
            'available_memory': 0,
            'cpu_count': 0,
            'cpu_cores': 0,
            'platform': sys.platform,
        }

        # 시스템 메모리 정보
        try:
            memory = psutil.virtual_memory()
            hardware_info.update({
                'total_memory': memory.total // (1024**3),
                'available_memory': memory.available // (1024**3),
                'cpu_count': psutil.cpu_count(),
                'cpu_cores': psutil.cpu_count(logical=False) or psutil.cpu_count(),
            })
        except Exception as e:
            logging.warning(f"Failed to get system memory info: {e}")

        # CUDA 정보
        try:
            import torch
            if torch.cuda.is_available():
                hardware_info['cuda_available'] = True
                hardware_info['cuda_device_count'] = torch.cuda.device_count()

                for i in range(torch.cuda.device_count()):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        gpu_memory = props.total_memory // (1024**3)
                        hardware_info[f'gpu_{i}_memory'] = gpu_memory
                        hardware_info[f'gpu_{i}_name'] = props.name
                        hardware_info[f'gpu_{i}_compute_capability'] = f"{props.major}.{props.minor}"
                    except Exception as e:
                        logging.warning(f"Failed to get GPU {i} info: {e}")

            # Apple Silicon 지원
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                hardware_info['mps_available'] = True

        except ImportError:
            logging.warning("PyTorch not available for hardware detection")
        except Exception as e:
            logging.warning(f"Failed to detect CUDA: {e}")

        return hardware_info

    @classmethod
    def get_recommended_config(cls, model_size: str, hardware_info: Optional[Dict] = None) -> ModelConfig:
        """하드웨어에 최적화된 안전한 설정 추천"""
        if hardware_info is None:
            hardware_info = cls.detect_hardware()

        # 기본 안전 설정
        config = ModelConfig(
            name="recommended",
            model_path="",
            device="auto",
            dtype="float16",
            load_in_4bit=True,
            trust_remote_code=False
        )

        # 모델 크기별 조정
        model_size_lower = model_size.lower()

        if hardware_info['cuda_available']:
            total_gpu_memory = sum(
                hardware_info.get(f'gpu_{i}_memory', 0)
                for i in range(hardware_info['cuda_device_count'])
            )
            config = cls._configure_for_cuda(config, model_size_lower, total_gpu_memory)
        elif hardware_info['mps_available']:
            config = cls._configure_for_mps(config, model_size_lower)
        else:
            config = cls._configure_for_cpu(config, model_size_lower)

        return config

    @classmethod
    def _configure_for_cuda(cls, config: ModelConfig, model_size: str, total_memory: int) -> ModelConfig:
        """CUDA 환경 설정"""
        if '0.5b' in model_size or '1b' in model_size:
            config.load_in_4bit = False
            config.dtype = "float16"
        elif '3b' in model_size or '7b' in model_size:
            if total_memory >= 16:
                config.load_in_4bit = False
                config.dtype = "float16"
            else:
                config.load_in_4bit = True
        elif '13b' in model_size or '14b' in model_size:
            if total_memory >= 32:
                config.load_in_4bit = True
            else:
                config.load_in_4bit = True
                config.cpu_offload = True
        elif '30b' in model_size or '70b' in model_size:
            if total_memory >= 80:
                config.tensor_parallel_size = min(4, total_memory // 20)
                config.load_in_4bit = True
            else:
                raise ValueError(f"Insufficient GPU memory for {model_size} model")

        return config

    @classmethod
    def _configure_for_mps(cls, config: ModelConfig, model_size: str) -> ModelConfig:
        """Apple Silicon MPS 설정"""
        config.device = "mps"
        config.dtype = "float16"

        # MPS는 양자화를 지원하지 않음
        config.load_in_4bit = False
        config.load_in_8bit = False

        # 큰 모델은 CPU 오프로드 필요할 수 있음
        if '30b' in model_size or '70b' in model_size:
            config.cpu_offload = True

        return config

    @classmethod
    def _configure_for_cpu(cls, config: ModelConfig, model_size: str) -> ModelConfig:
        """CPU 전용 설정"""
        config.device = "cpu"
        config.dtype = "float32"
        config.load_in_4bit = False
        config.load_in_8bit = False

        # CPU에서는 작은 모델만 권장
        if '30b' in model_size or '70b' in model_size:
            logging.warning(f"Large model {model_size} not recommended for CPU inference")

        return config

class SafetyChecker:
    """안전성 검사기"""

    @staticmethod
    def check_model_config(config: ModelConfig) -> List[str]:
        """모델 설정 안전성 검사"""
        warnings = []

        # 메모리 안전성 검사
        memory_warnings = SafetyChecker.check_memory_safety(config)
        warnings.extend(memory_warnings)

        # 보안 검사
        security_warnings = SafetyChecker.check_security(config)
        warnings.extend(security_warnings)

        # 호환성 검사
        compatibility_warnings = SafetyChecker.check_compatibility(config)
        warnings.extend(compatibility_warnings)

        return warnings

    @staticmethod
    def check_memory_safety(config: ModelConfig) -> List[str]:
        """메모리 안전성 검사"""
        warnings = []

        try:
            import torch
            if config.device == "cuda" and torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                estimated_usage = SafetyChecker._estimate_memory_usage(config)

                if estimated_usage > available_memory * 0.9:
                    warnings.append(
                        f"Estimated memory usage ({estimated_usage:.1f}GB) exceeds 90% "
                        f"of available GPU memory ({available_memory:.1f}GB)"
                    )
                    warnings.append("Consider enabling quantization or using a smaller model")

                if estimated_usage > available_memory:
                    warnings.append("Estimated memory usage exceeds available GPU memory")

        except Exception as e:
            warnings.append(f"Failed to check memory safety: {e}")

        return warnings

    @staticmethod
    def check_security(config: ModelConfig) -> List[str]:
        """보안 검사"""
        warnings = []

        if config.trust_remote_code:
            warnings.append(
                "trust_remote_code=True allows execution of arbitrary code. "
                "Only use with trusted models."
            )

        # 모델 경로 검사
        if config.model_path and not config.model_path.startswith(('huggingface.co', 'hf.co')):
            if '/' in config.model_path and not config.model_path.startswith('./'):
                warnings.append(
                    "Custom model path detected. Ensure the model source is trusted."
                )

        return warnings

    @staticmethod
    def check_compatibility(config: ModelConfig) -> List[str]:
        """호환성 검사"""
        warnings = []

        # 양자화 호환성
        if config.load_in_4bit and config.load_in_8bit:
            warnings.append("Cannot use both 4-bit and 8-bit quantization simultaneously")

        # MPS 호환성
        if config.device == "mps" and (config.load_in_4bit or config.load_in_8bit):
            warnings.append("MPS device does not support quantization")

        # vLLM 호환성
        if config.model_type == "vllm" and config.device == "cpu":
            warnings.append("vLLM requires CUDA-capable GPU")

        return warnings

    @staticmethod
    def _estimate_memory_usage(config: ModelConfig) -> float:
        """메모리 사용량 추정"""
        # 간단한 모델 크기 기반 추정
        model_size_estimates = {
            '0.5b': 1.0, '1b': 2.0, '1.5b': 3.0,
            '3b': 6.0, '7b': 14.0, '8b': 16.0,
            '13b': 26.0, '14b': 28.0,
            '30b': 60.0, '32b': 64.0,
            '70b': 140.0, '72b': 144.0
        }

        base_memory = 14.0  # 기본값
        model_path_lower = config.model_path.lower()

        for size, memory in model_size_estimates.items():
            if size in model_path_lower:
                base_memory = memory
                break

        # 양자화 적용
        if config.load_in_4bit:
            return base_memory * 0.5
        elif config.load_in_8bit:
            return base_memory * 0.75
        else:
            return base_memory

class ConfigManager:
    """안전한 설정 관리자"""

    def __init__(self, config_file: str = "llm_config.json"):
        self.config_file = Path(config_file)
        self.model_configs: Dict[str, ModelConfig] = {}
        self.test_configs: Dict[str, TestConfig] = {}
        self.optimization_config = OptimizationConfig()

        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

        # 설정 로드
        self.load_config()

    def load_config(self):
        """안전한 설정 로드"""
        with self._lock:
            if self.config_file.exists():
                try:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)

                    self._load_model_configs(config_data.get('models', {}))
                    self._load_test_configs(config_data.get('tests', {}))
                    self._load_optimization_config(config_data.get('optimization', {}))

                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in config file: {e}")
                    self.create_default_config()
                except Exception as e:
                    self.logger.error(f"Failed to load config: {e}")
                    self.create_default_config()
            else:
                self.create_default_config()

    def _load_model_configs(self, models_data: Dict[str, Any]):
        """모델 설정 로드"""
        for name, cfg in models_data.items():
            try:
                self.model_configs[name] = ModelConfig.from_dict(cfg)
            except Exception as e:
                self.logger.warning(f"Failed to load model config {name}: {e}")
                # 기본 설정으로 대체
                self.model_configs[name] = self._create_fallback_model_config(name)

    def _load_test_configs(self, tests_data: Dict[str, Any]):
        """테스트 설정 로드"""
        for name, cfg in tests_data.items():
            try:
                self.test_configs[name] = TestConfig(**cfg)
            except Exception as e:
                self.logger.warning(f"Failed to load test config {name}: {e}")

    def _load_optimization_config(self, opt_data: Dict[str, Any]):
        """최적화 설정 로드"""
        try:
            self.optimization_config = OptimizationConfig(**opt_data)
        except Exception as e:
            self.logger.warning(f"Failed to load optimization config: {e}")
            self.optimization_config = OptimizationConfig()

    def create_default_config(self):
        """안전한 기본 설정 생성"""
        hardware_info = HardwareDetector.detect_hardware()

        default_models = {
            "llama2-7b": {
                "name": "llama2-7b",
                "model_path": "meta-llama/Llama-2-7b-chat-hf",
                "model_type": "transformers",
                "device": "auto",
                "dtype": "float16",
                "load_in_4bit": True,
                "trust_remote_code": False
            },
            "qwen2.5-7b": {
                "name": "qwen2.5-7b",
                "model_path": "Qwen/Qwen2.5-7B-Instruct",
                "model_type": "transformers",
                "device": "auto",
                "dtype": "float16",
                "load_in_4bit": True,
                "trust_remote_code": False
            }
        }

        default_tests = {
            "korean_math": {
                "dataset_name": "Korean Math",
                "dataset_path": "data/korean_math.json",
                "num_samples": 50
            },
            "korean_qa": {
                "dataset_name": "Korean QA",
                "dataset_path": "data/korean_qa.json",
                "num_samples": 50
            }
        }

        default_optimization = {
            "enable_torch_compile": False,
            "enable_bettertransformer": False,
            "dynamic_batching": False,
            "max_batch_size": 4,
            "max_sequence_length": 1024,
            "auto_cleanup": True
        }

        config_data = {
            "models": default_models,
            "tests": default_tests,
            "optimization": default_optimization,
            "hardware_info": hardware_info
        }

        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        # 설정 다시 로드
        self._load_model_configs(default_models)
        self._load_test_configs(default_tests)
        self._load_optimization_config(default_optimization)

    def _create_fallback_model_config(self, model_name: str) -> ModelConfig:
        """대체 모델 설정 생성"""
        return ModelConfig(
            name=model_name,
            model_path=model_name,
            model_type="transformers",
            device="auto",
            dtype="float16",
            load_in_4bit=True,
            trust_remote_code=False
        )

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """안전한 모델 설정 반환"""
        with self._lock:
            if model_name not in self.model_configs:
                self.logger.warning(f"Model {model_name} not found in configuration")
                return None

            config = self.model_configs[model_name]

            # 안전성 검사
            warnings = SafetyChecker.check_model_config(config)
            if warnings:
                self.logger.warning(f"Model '{model_name}' has safety warnings:")
                for warning in warnings:
                    self.logger.warning(f"  - {warning}")

                # 자동 수정 시도
                config = self._auto_fix_config(config, warnings)

            return config

    def _auto_fix_config(self, config: ModelConfig, warnings: List[str]) -> ModelConfig:
        """설정 자동 수정"""
        # 딕셔너리로 변환하여 수정
        config_dict = config.to_dict()

        for warning in warnings:
            if "memory usage exceeds" in warning.lower():
                config_dict['load_in_4bit'] = True
                config_dict['gpu_memory_utilization'] = 0.7
            elif "cannot use both" in warning.lower():
                config_dict['load_in_8bit'] = False  # 4-bit 우선
            elif "mps" in warning.lower() and "quantization" in warning.lower():
                config_dict['load_in_4bit'] = False
                config_dict['load_in_8bit'] = False

        return ModelConfig.from_dict(config_dict)

    def add_model_config(self, name: str, config: ModelConfig):
        """모델 설정 추가"""
        with self._lock:
            # 안전성 검사
            warnings = SafetyChecker.check_model_config(config)
            if warnings:
                for warning in warnings:
                    self.logger.warning(f"Adding model '{name}' with warning: {warning}")

            self.model_configs[name] = config
            self.save_config()

    def remove_model_config(self, name: str) -> bool:
        """모델 설정 제거"""
        with self._lock:
            if name in self.model_configs:
                del self.model_configs[name]
                self.save_config()
                return True
            return False

    def save_config(self):
        """설정 저장"""
        with self._lock:
            try:
                config_data = {
                    "models": {name: config.to_dict() for name, config in self.model_configs.items()},
                    "tests": {name: asdict(config) for name, config in self.test_configs.items()},
                    "optimization": asdict(self.optimization_config),
                    "hardware_info": HardwareDetector.detect_hardware()
                }

                # 백업 생성
                if self.config_file.exists():
                    backup_path = self.config_file.with_suffix('.json.bak')
                    self.config_file.rename(backup_path)

                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)

                self.logger.info(f"Configuration saved to {self.config_file}")

            except Exception as e:
                self.logger.error(f"Failed to save configuration: {e}")
                # 백업 복원
                backup_path = self.config_file.with_suffix('.json.bak')
                if backup_path.exists():
                    backup_path.rename(self.config_file)

    def validate_all_configs(self) -> Dict[str, List[str]]:
        """모든 설정 검증"""
        validation_results = {}

        for name, config in self.model_configs.items():
            warnings = SafetyChecker.check_model_config(config)
            if warnings:
                validation_results[name] = warnings

        return validation_results

    @contextmanager
    def temp_model_config(self, name: str, config: ModelConfig):
        """임시 모델 설정 (컨텍스트 매니저)"""
        original_config = self.model_configs.get(name)
        try:
            self.model_configs[name] = config
            yield config
        finally:
            if original_config is not None:
                self.model_configs[name] = original_config
            elif name in self.model_configs:
                del self.model_configs[name]

class ResourceManager:
    """강화된 리소스 관리자"""

    def __init__(self):
        self._active_models: Dict[str, weakref.ref] = {}
        self._memory_threshold = 0.85  # 85%로 낮춤
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._shutdown = False
        self.logger = logging.getLogger(__name__)

        # 자동 정리 스레드 시작
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """자동 정리 스레드 시작"""
        def cleanup_loop():
            while not self._shutdown:
                try:
                    self._periodic_cleanup()
                    import time
                    time.sleep(60)  # 1분마다 정리
                except Exception as e:
                    self.logger.error(f"Cleanup thread error: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def register_model(self, name: str, model_interface):
        """모델 등록"""
        with self._lock:
            self._active_models[name] = weakref.ref(
                model_interface,
                lambda ref: self._on_model_deleted(name, ref)
            )

    def _on_model_deleted(self, name: str, ref: weakref.ref):
        """모델 삭제 시 콜백"""
        with self._lock:
            if name in self._active_models and self._active_models[name] is ref:
                del self._active_models[name]
        self.logger.info(f"Model {name} automatically cleaned up")

    def cleanup_model(self, name: str):
        """명시적 모델 정리"""
        with self._lock:
            if name in self._active_models:
                model_ref = self._active_models[name]
                model = model_ref()
                if model is not None:
                    self._cleanup_model_instance(model)
                del self._active_models[name]

    def _cleanup_model_instance(self, model_instance):
        """모델 인스턴스 정리"""
        try:
            # 모델을 CPU로 이동
            if hasattr(model_instance, 'model') and hasattr(model_instance.model, 'cpu'):
                model_instance.model.cpu()

            # 메모리 해제
            if hasattr(model_instance, 'model'):
                del model_instance.model

            # CUDA 캐시 정리
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass

            # 가비지 컬렉션
            import gc
            gc.collect()

        except Exception as e:
            self.logger.error(f"Error during model cleanup: {e}")

    def _periodic_cleanup(self):
        """주기적 정리"""
        with self._lock:
            # 죽은 참조 제거
            dead_refs = []
            for name, ref in self._active_models.items():
                if ref() is None:
                    dead_refs.append(name)

            for name in dead_refs:
                del self._active_models[name]

            # 메모리 사용량 확인
            memory_usage = self.get_memory_usage()
            if memory_usage.get('gpu_utilization', 0) > self._memory_threshold:
                self.logger.warning(f"High GPU memory usage: {memory_usage['gpu_utilization']:.1%}")
                self._emergency_cleanup()

    def _emergency_cleanup(self):
        """긴급 메모리 정리"""
        self.logger.info("Emergency memory cleanup initiated")

        # 모든 활성 모델 정리
        model_names = list(self._active_models.keys())
        for name in model_names:
            self.cleanup_model(name)

        # 강제 가비지 컬렉션
        import gc
        gc.collect()

        # CUDA 메모리 완전 정리
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
        except:
            pass

    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 조회"""
        usage = {}

        # 시스템 메모리
        try:
            memory = psutil.virtual_memory()
            usage['system_memory_percent'] = memory.percent
            usage['system_memory_gb'] = memory.used / 1024**3
        except:
            pass

        # GPU 메모리
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3

                    usage[f'gpu_{i}_allocated_gb'] = allocated
                    usage[f'gpu_{i}_reserved_gb'] = reserved
                    usage[f'gpu_{i}_total_gb'] = total
                    usage[f'gpu_{i}_utilization'] = allocated / total if total > 0 else 0

                # 전체 GPU 사용률
                total_allocated = sum(usage.get(f'gpu_{i}_allocated_gb', 0) for i in range(torch.cuda.device_count()))
                total_capacity = sum(usage.get(f'gpu_{i}_total_gb', 0) for i in range(torch.cuda.device_count()))
                usage['gpu_utilization'] = total_allocated / total_capacity if total_capacity > 0 else 0
        except:
            pass

        return usage

    def shutdown(self):
        """리소스 매니저 종료"""
        self._shutdown = True
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

        # 모든 모델 정리
        model_names = list(self._active_models.keys())
        for name in model_names:
            self.cleanup_model(name)

# 전역 리소스 매니저 인스턴스
_global_resource_manager = None
_resource_manager_lock = threading.Lock()

def get_resource_manager() -> ResourceManager:
    """전역 리소스 매니저 반환"""
    global _global_resource_manager

    with _resource_manager_lock:
        if _global_resource_manager is None:
            _global_resource_manager = ResourceManager()

    return _global_resource_manager

# 정리 함수
def cleanup_resources():
    """전역 리소스 정리"""
    global _global_resource_manager

    with _resource_manager_lock:
        if _global_resource_manager is not None:
            _global_resource_manager.shutdown()
            _global_resource_manager = None

    # 환경 변수 복원
    SafeEnvironmentManager.restore_environment()

# exit 핸들러 등록
import atexit
atexit.register(cleanup_resources)

# 사용 예시
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== 안전한 설정 관리 시스템 테스트 ===")

    try:
        # 설정 매니저 생성
        config_manager = ConfigManager()

        # 하드웨어 정보 출력
        hardware_info = HardwareDetector.detect_hardware()
        print("하드웨어 정보:")
        for key, value in hardware_info.items():
            print(f"  {key}: {value}")

        # 모델 설정 검증
        validation_results = config_manager.validate_all_configs()
        if validation_results:
            print("\n설정 검증 결과:")
            for model_name, warnings in validation_results.items():
                print(f"  {model_name}:")
                for warning in warnings:
                    print(f"    - {warning}")
        else:
            print("\n✅ 모든 설정이 안전합니다")

        # 추천 설정 테스트
        print("\n추천 설정:")
        recommended = HardwareDetector.get_recommended_config("7b")
        print(f"  Device: {recommended.device}")
        print(f"  Dtype: {recommended.dtype}")
        print(f"  4-bit quantization: {recommended.load_in_4bit}")

        # 리소스 매니저 테스트
        resource_manager = get_resource_manager()
        memory_usage = resource_manager.get_memory_usage()
        print(f"\n메모리 사용량:")
        for key, value in memory_usage.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        print("\n✅ 안전한 설정 시스템 테스트 완료")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_resources()