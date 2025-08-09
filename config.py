"""
오픈소스 LLM 추론 성능 최적화를 위한 설정 및 파라미터 관리
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import os

@dataclass
class ModelConfig:
    """모델 설정 클래스"""
    name: str
    model_path: str  # 로컬 모델 경로 또는 HuggingFace ID
    model_type: str = "transformers"  # transformers, vllm, ollama, tgi
    device: str = "auto"  # cuda, cpu, auto
    dtype: str = "float16"  # float16, bfloat16, int8, int4
    max_memory: Optional[str] = None  # "8GB", "auto"
    trust_remote_code: bool = False
    use_flash_attention: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # vLLM 특화 설정
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9

    # TGI 특화 설정
    max_batch_prefill_tokens: Optional[int] = None
    max_batch_total_tokens: Optional[int] = None

    # Ollama 특화 설정
    base_url: str = "http://localhost:11434"

    # 양자화 설정
    quantization_config: Optional[Dict[str, Any]] = None

@dataclass
class InferenceParams:
    """추론 파라미터 클래스"""
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

    # 성능 최적화 파라미터
    use_cache: bool = True
    batch_size: int = 1
    num_return_sequences: int = 1

@dataclass
class OptimizationConfig:
    """최적화 설정 클래스"""
    enable_torch_compile: bool = False
    enable_bettertransformer: bool = False
    enable_ipex: bool = False  # Intel Extension for PyTorch
    enable_deepspeed: bool = False
    deepspeed_config: Optional[str] = None

    # 메모리 최적화
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    disk_offload: bool = False

    # 배치 처리 최적화
    dynamic_batching: bool = True
    max_batch_size: int = 32
    max_sequence_length: int = 2048

@dataclass
class TestConfig:
    """테스트 설정 클래스"""
    dataset_name: str
    dataset_path: str
    output_dir: str = "results"
    num_samples: Optional[int] = None
    random_seed: int = 42
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True

class ConfigManager:
    """설정 관리 클래스"""

    def __init__(self, config_file: str = "llm_config.json"):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        """설정 파일 로드"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                self.model_configs = {
                    name: ModelConfig(**cfg)
                    for name, cfg in config_data.get('models', {}).items()
                }
                self.test_configs = {
                    name: TestConfig(**cfg)
                    for name, cfg in config_data.get('tests', {}).items()
                }
                self.optimization_config = OptimizationConfig(
                    **config_data.get('optimization', {})
                )
        else:
            self.create_default_config()

    def create_default_config(self):
        """기본 설정 생성"""
        default_config = {
            "models": {
                "llama2-7b": {
                    "name": "llama2-7b",
                    "model_path": "meta-llama/Llama-2-7b-chat-hf",
                    "model_type": "transformers",
                    "device": "auto",
                    "dtype": "float16",
                    "use_flash_attention": True,
                    "trust_remote_code": False
                },
                "mistral-7b": {
                    "name": "mistral-7b",
                    "model_path": "mistralai/Mistral-7B-Instruct-v0.2",
                    "model_type": "vllm",
                    "device": "cuda",
                    "dtype": "float16",
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.9
                },
                "gemma-7b": {
                    "name": "gemma-7b",
                    "model_path": "google/gemma-7b-it",
                    "model_type": "transformers",
                    "device": "auto",
                    "dtype": "bfloat16",
                    "load_in_4bit": True
                },
                "qwen-7b": {
                    "name": "qwen-7b",
                    "model_path": "Qwen/Qwen1.5-7B-Chat",
                    "model_type": "transformers",
                    "device": "auto",
                    "dtype": "float16",
                    "trust_remote_code": True
                }
            },
            "tests": {
                "math_reasoning": {
                    "dataset_name": "GSM8K",
                    "dataset_path": "data/gsm8k.json",
                    "num_samples": 100
                },
                "commonsense": {
                    "dataset_name": "HellaSwag",
                    "dataset_path": "data/hellaswag.json",
                    "num_samples": 200
                },
                "korean_qa": {
                    "dataset_name": "KoreanQA",
                    "dataset_path": "data/korean_qa.json",
                    "num_samples": 150
                }
            },
            "optimization": {
                "enable_torch_compile": False,
                "enable_bettertransformer": True,
                "dynamic_batching": True,
                "max_batch_size": 16,
                "max_sequence_length": 2048
            }
        }

        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)

        self.load_config()

    def get_param_grid(self) -> Dict[str, List]:
        """파라미터 그리드 반환"""
        return {
            'temperature': [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
            'top_p': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            'top_k': [1, 5, 10, 20, 40, 50],
            'max_new_tokens': [50, 100, 200, 512, 1024],
            'repetition_penalty': [1.0, 1.05, 1.1, 1.15, 1.2]
        }

    def get_optimization_param_grid(self) -> Dict[str, List]:
        """최적화 파라미터 그리드 반환"""
        return {
            'batch_size': [1, 2, 4, 8, 16, 32],
            'use_cache': [True, False],
            'do_sample': [True, False],
            'num_beams': [1, 2, 4],
            'max_sequence_length': [512, 1024, 2048, 4096]
        }

    def get_reasoning_params(self) -> List[InferenceParams]:
        """추론 최적화된 파라미터 조합 반환"""
        return [
            InferenceParams(temperature=0.0, top_p=0.1, top_k=1, max_new_tokens=200, do_sample=False),
            InferenceParams(temperature=0.1, top_p=0.3, top_k=10, max_new_tokens=300, repetition_penalty=1.05),
            InferenceParams(temperature=0.2, top_p=0.5, top_k=20, max_new_tokens=400, repetition_penalty=1.1),
        ]

    def get_generation_params(self) -> List[InferenceParams]:
        """생성 최적화된 파라미터 조합 반환"""
        return [
            InferenceParams(temperature=0.7, top_p=0.9, top_k=40, max_new_tokens=512, repetition_penalty=1.1),
            InferenceParams(temperature=0.8, top_p=0.95, top_k=50, max_new_tokens=800, repetition_penalty=1.15),
            InferenceParams(temperature=1.0, top_p=1.0, top_k=50, max_new_tokens=1024, repetition_penalty=1.2),
        ]

    def get_model_families(self) -> Dict[str, List[str]]:
        """모델 패밀리별 그룹화"""
        families = {}
        for name, config in self.model_configs.items():
            if 'llama' in name.lower():
                families.setdefault('llama', []).append(name)
            elif 'mistral' in name.lower():
                families.setdefault('mistral', []).append(name)
            elif 'gemma' in name.lower():
                families.setdefault('gemma', []).append(name)
            elif 'qwen' in name.lower():
                families.setdefault('qwen', []).append(name)
            else:
                families.setdefault('other', []).append(name)
        return families

# 하드웨어 감지 유틸리티
class HardwareDetector:
    """하드웨어 사양 감지 및 최적화 설정 추천"""

    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """하드웨어 사양 감지"""
        import torch
        import psutil

        hardware_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'total_memory': psutil.virtual_memory().total // (1024**3),  # GB
            'available_memory': psutil.virtual_memory().available // (1024**3),  # GB
            'cpu_count': psutil.cpu_count(),
            'cpu_cores': psutil.cpu_count(logical=False)
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                hardware_info[f'gpu_{i}_memory'] = gpu_memory
                hardware_info[f'gpu_{i}_name'] = torch.cuda.get_device_properties(i).name

        return hardware_info

    @staticmethod
    def recommend_config(model_size: str) -> ModelConfig:
        """모델 크기에 따른 최적 설정 추천"""
        hardware = HardwareDetector.detect_hardware()

        # 기본 설정
        config = ModelConfig(
            name="recommended",
            model_path="",
            device="auto",
            dtype="float16"
        )

        # GPU 메모리에 따른 설정 조정
        if hardware['cuda_available']:
            total_gpu_memory = sum(
                hardware.get(f'gpu_{i}_memory', 0)
                for i in range(hardware['cuda_device_count'])
            )

            if model_size in ['7b', '8b']:
                if total_gpu_memory >= 16:
                    config.dtype = "float16"
                    config.load_in_4bit = False
                elif total_gpu_memory >= 8:
                    config.dtype = "float16"
                    config.load_in_4bit = True
                else:
                    config.dtype = "int8"
                    config.load_in_8bit = True

            elif model_size in ['13b', '14b']:
                if total_gpu_memory >= 32:
                    config.dtype = "float16"
                    config.tensor_parallel_size = 1
                elif total_gpu_memory >= 16:
                    config.dtype = "float16"
                    config.load_in_4bit = True
                else:
                    config.cpu_offload = True
                    config.load_in_8bit = True

            elif model_size in ['70b', '72b']:
                if total_gpu_memory >= 160:
                    config.tensor_parallel_size = min(4, hardware['cuda_device_count'])
                elif total_gpu_memory >= 80:
                    config.tensor_parallel_size = min(2, hardware['cuda_device_count'])
                    config.load_in_4bit = True
                else:
                    config.cpu_offload = True
                    config.disk_offload = True
        else:
            # CPU only
            config.device = "cpu"
            config.dtype = "float32"
            config.load_in_8bit = True

        return config

# 사용 예시
if __name__ == "__main__":
    config_manager = ConfigManager()

    # 하드웨어 정보 출력
    hardware_info = HardwareDetector.detect_hardware()
    print("하드웨어 정보:")
    for key, value in hardware_info.items():
        print(f"  {key}: {value}")

    # 모델 설정 출력
    print("\n모델 설정:")
    for name, config in config_manager.model_configs.items():
        print(f"  {name}: {config.model_path} ({config.model_type})")

    # 모델 패밀리 출력
    print("\n모델 패밀리:")
    families = config_manager.get_model_families()
    for family, models in families.items():
        print(f"  {family}: {', '.join(models)}")

    # 추천 설정
    print("\n7B 모델 추천 설정:")
    recommended = HardwareDetector.recommend_config("7b")
    print(f"  Device: {recommended.device}")
    print(f"  Dtype: {recommended.dtype}")
    print(f"  4-bit: {recommended.load_in_4bit}")
    print(f"  8-bit: {recommended.load_in_8bit}")