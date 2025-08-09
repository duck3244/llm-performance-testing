"""
안전한 오픈소스 LLM 추론 성능 최적화를 위한 설정 및 파라미터 관리
모든 문제점이 해결된 버전
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import os
import logging
import warnings
from pathlib import Path

@dataclass
class ModelConfig:
    """모델 설정 클래스 - 안전성 강화"""
    name: str
    model_path: str
    model_type: str = "transformers"
    device: str = "auto"
    dtype: str = "float16"
    max_memory: Optional[str] = None
    trust_remote_code: bool = False
    use_flash_attention: bool = False  # 기본값 변경
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # vLLM 특화 설정
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.8  # 더 안전한 값

    # TGI 특화 설정
    max_batch_prefill_tokens: Optional[int] = None
    max_batch_total_tokens: Optional[int] = None

    # Ollama 특화 설정
    base_url: str = "http://localhost:11434"

    # 양자화 설정
    quantization_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """초기화 후 검증"""
        # 안전한 기본값 설정
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        # 메모리 부족 시 자동 양자화
        if self.device == "cuda" and not self.load_in_4bit and not self.load_in_8bit:
            try:
                import torch
                if torch.cuda.is_available():
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    if total_memory < 16:  # 16GB 미만
                        self.load_in_4bit = True
                        logging.info(f"Auto-enabled 4-bit quantization for GPU with {total_memory:.1f}GB memory")
            except Exception:
                pass

@dataclass
class InferenceParams:
    """추론 파라미터 클래스 - 안전한 범위"""
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

        # 논리 검증
        if self.min_new_tokens > self.max_new_tokens:
            self.min_new_tokens = 1

        # 샘플링 일관성 확인
        if self.temperature == 0.0:
            self.do_sample = False

@dataclass
class OptimizationConfig:
    """최적화 설정 클래스 - 안전성 우선"""
    enable_torch_compile: bool = False  # 기본값 변경
    enable_bettertransformer: bool = False  # 기본값 변경
    enable_ipex: bool = False
    enable_deepspeed: bool = False
    deepspeed_config: Optional[str] = None

    # 메모리 최적화
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    disk_offload: bool = False

    # 배치 처리 최적화
    dynamic_batching: bool = False  # 기본값 변경
    max_batch_size: int = 8  # 더 안전한 값
    max_sequence_length: int = 2048

@dataclass
class TestConfig:
    """테스트 설정 클래스"""
    dataset_name: str
    dataset_path: str
    output_dir: str = "results"
    num_samples: Optional[int] = None
    random_seed: int = 42
    num_workers: int = 1  # 안전한 값
    prefetch_factor: int = 1  # 안전한 값
    pin_memory: bool = False  # 기본값 변경

class ConfigValidator:
    """설정 검증 클래스"""

    @staticmethod
    def validate_model_config(config: ModelConfig) -> List[str]:
        """모델 설정 검증"""
        issues = []

        if not config.model_path:
            issues.append("model_path is required")

        if config.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    issues.append("CUDA not available but device set to 'cuda'")
            except ImportError:
                issues.append("PyTorch not available for CUDA check")

        if config.load_in_4bit and config.load_in_8bit:
            issues.append("Cannot use both 4-bit and 8-bit quantization")

        if config.max_memory:
            try:
                if config.max_memory.endswith('GB'):
                    memory_gb = float(config.max_memory[:-2])
                    if memory_gb <= 0:
                        issues.append("max_memory must be positive")
            except ValueError:
                issues.append("Invalid max_memory format")

        # vLLM 특화 검증
        if config.model_type == "vllm":
            if config.tensor_parallel_size < 1:
                issues.append("tensor_parallel_size must be >= 1")
            if not 0.1 <= config.gpu_memory_utilization <= 1.0:
                issues.append("gpu_memory_utilization must be between 0.1 and 1.0")

        return issues

    @staticmethod
    def validate_inference_params(params: InferenceParams) -> List[str]:
        """추론 파라미터 검증"""
        issues = []

        if not 0 <= params.temperature <= 2.0:
            issues.append("temperature must be between 0 and 2.0")

        if not 0 < params.top_p <= 1.0:
            issues.append("top_p must be between 0 and 1.0")

        if params.top_k <= 0:
            issues.append("top_k must be positive")

        if params.max_new_tokens <= 0:
            issues.append("max_new_tokens must be positive")

        if params.max_new_tokens > 2048:
            issues.append("max_new_tokens should not exceed 2048 for safety")

        if not 1.0 <= params.repetition_penalty <= 2.0:
            issues.append("repetition_penalty should be between 1.0 and 2.0")

        return issues

class RobustConfigManager:
    """강화된 설정 관리자"""

    def __init__(self, config_file: str = "llm_config.json"):
        self.config_file = config_file
        self.validator = ConfigValidator()
        self.logger = logging.getLogger(__name__)

        self.load_config()
        self._validate_all_configs()

    def load_config(self):
        """설정 파일 로드 - 안전한 방식"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                self.model_configs = {}
                self.test_configs = {}

                # 모델 설정 로드
                for name, cfg in config_data.get('models', {}).items():
                    try:
                        self.model_configs[name] = ModelConfig(**cfg)
                    except Exception as e:
                        self.logger.warning(f"Failed to load model config {name}: {e}")
                        # 기본 설정으로 대체
                        self.model_configs[name] = self._create_fallback_model_config(name)

                # 테스트 설정 로드
                for name, cfg in config_data.get('tests', {}).items():
                    try:
                        self.test_configs[name] = TestConfig(**cfg)
                    except Exception as e:
                        self.logger.warning(f"Failed to load test config {name}: {e}")

                # 최적화 설정 로드
                try:
                    self.optimization_config = OptimizationConfig(
                        **config_data.get('optimization', {})
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to load optimization config: {e}")
                    self.optimization_config = OptimizationConfig()

            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in config file: {e}")
                self.create_default_config()
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")
                self.create_default_config()
        else:
            self.create_default_config()

    def create_default_config(self):
        """기본 설정 생성 - 안전한 기본값"""
        default_config = {
            "models": {
                "llama2-7b": {
                    "name": "llama2-7b",
                    "model_path": "meta-llama/Llama-2-7b-chat-hf",
                    "model_type": "transformers",
                    "device": "auto",
                    "dtype": "float16",
                    "load_in_4bit": True,  # 안전한 기본값
                    "trust_remote_code": False
                },
                "mistral-7b": {
                    "name": "mistral-7b",
                    "model_path": "mistralai/Mistral-7B-Instruct-v0.2",
                    "model_type": "transformers",  # vLLM 대신 안전한 기본값
                    "device": "auto",
                    "dtype": "float16",
                    "load_in_4bit": True,
                    "gpu_memory_utilization": 0.8
                },
                "gemma-7b": {
                    "name": "gemma-7b",
                    "model_path": "google/gemma-7b-it",
                    "model_type": "transformers",
                    "device": "auto",
                    "dtype": "float16",  # bfloat16 대신
                    "load_in_4bit": True
                },
                "qwen-7b": {
                    "name": "qwen-7b",
                    "model_path": "Qwen/Qwen1.5-7B-Chat",
                    "model_type": "transformers",
                    "device": "auto",
                    "dtype": "float16",
                    "load_in_4bit": True,
                    "trust_remote_code": False  # 안전한 기본값
                }
            },
            "tests": {
                "korean_math": {
                    "dataset_name": "Korean Math",
                    "dataset_path": "data/korean_math.json",
                    "num_samples": 50
                },
                "korean_qa": {
                    "dataset_name": "Korean QA",
                    "dataset_path": "data/korean_qa.json",
                    "num_samples": 50
                },
                "korean_reasoning": {
                    "dataset_name": "Korean Reasoning",
                    "dataset_path": "data/korean_reasoning.json",
                    "num_samples": 50
                }
            },
            "optimization": {
                "enable_torch_compile": False,
                "enable_bettertransformer": False,
                "dynamic_batching": False,
                "max_batch_size": 4,  # 안전한 값
                "max_sequence_length": 1024  # 안전한 값
            }
        }

        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)

        self.load_config()

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

    def _validate_all_configs(self):
        """모든 설정 검증"""
        all_issues = []

        # 모델 설정 검증
        for name, config in self.model_configs.items():
            issues = self.validator.validate_model_config(config)
            if issues:
                all_issues.extend([f"Model '{name}': {issue}" for issue in issues])

        # 이슈가 있으면 경고
        if all_issues:
            warnings.warn(f"Configuration issues found:\n" + "\n".join(all_issues))

    def get_safe_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """안전한 모델 설정 반환"""
        if model_name not in self.model_configs:
            self.logger.warning(f"Model {model_name} not found in configuration")
            return None

        config = self.model_configs[model_name]
        issues = self.validator.validate_model_config(config)

        if issues:
            self.logger.warning(f"Model '{model_name}' has configuration issues: {issues}")
            # 문제가 있어도 일단 반환하고 자동 수정 시도
            return self._auto_fix_config(config, issues)

        return config

    def _auto_fix_config(self, config: ModelConfig, issues: List[str]) -> ModelConfig:
        """설정 자동 수정"""
        fixed_config = config

        for issue in issues:
            if "CUDA not available" in issue:
                fixed_config.device = "cpu"
                fixed_config.dtype = "float32"
            elif "Cannot use both 4-bit and 8-bit" in issue:
                fixed_config.load_in_8bit = False  # 4-bit 우선
            elif "max_memory must be positive" in issue:
                fixed_config.max_memory = None

        return fixed_config

    def create_fallback_config(self, model_name: str) -> ModelConfig:
        """대체 설정 생성"""
        hardware_info = HardwareDetector.detect_hardware()

        device = "cuda" if hardware_info['cuda_available'] else "cpu"
        dtype = "float16" if hardware_info['cuda_available'] else "float32"

        return ModelConfig(
            name=model_name,
            model_path=model_name,
            model_type="transformers",
            device=device,
            dtype=dtype,
            load_in_4bit=hardware_info.get('gpu_0_memory', 0) < 16,
            trust_remote_code=False
        )

    def get_param_grid(self) -> Dict[str, List]:
        """안전한 파라미터 그리드 반환"""
        return {
            'temperature': [0.0, 0.1, 0.3, 0.7],  # 범위 축소
            'top_p': [0.1, 0.5, 0.9],
            'top_k': [1, 10, 50],
            'max_new_tokens': [100, 200, 300],  # 범위 축소
            'repetition_penalty': [1.0, 1.1]
        }

    def get_optimization_param_grid(self) -> Dict[str, List]:
        """최적화 파라미터 그리드 반환"""
        return {
            'batch_size': [1, 2, 4],  # 안전한 범위
            'use_cache': [True],  # 안전한 값만
            'do_sample': [True, False],
            'num_beams': [1, 2],  # 범위 축소
            'max_sequence_length': [512, 1024]  # 안전한 범위
        }

    def get_reasoning_params(self) -> List[InferenceParams]:
        """추론 최적화된 안전한 파라미터 조합"""
        return [
            InferenceParams(
                temperature=0.0,
                top_p=0.1,
                top_k=1,
                max_new_tokens=200,
                do_sample=False
            ),
            InferenceParams(
                temperature=0.1,
                top_p=0.3,
                top_k=10,
                max_new_tokens=300,
                repetition_penalty=1.05
            ),
            InferenceParams(
                temperature=0.2,
                top_p=0.5,
                top_k=20,
                max_new_tokens=400,
                repetition_penalty=1.1
            ),
        ]

    def get_generation_params(self) -> List[InferenceParams]:
        """생성 최적화된 안전한 파라미터 조합"""
        return [
            InferenceParams(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                max_new_tokens=512,
                repetition_penalty=1.1
            ),
            InferenceParams(
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                max_new_tokens=600,  # 축소
                repetition_penalty=1.15
            ),
        ]

    def get_model_families(self) -> Dict[str, List[str]]:
        """모델 패밀리별 그룹화"""
        families = {}
        for name, config in self.model_configs.items():
            family = self._detect_model_family(name)
            families.setdefault(family, []).append(name)
        return families

    def _detect_model_family(self, model_name: str) -> str:
        """모델 패밀리 감지"""
        name_lower = model_name.lower()
        if 'llama' in name_lower:
            return 'llama'
        elif 'mistral' in name_lower:
            return 'mistral'
        elif 'gemma' in name_lower:
            return 'gemma'
        elif 'qwen' in name_lower:
            return 'qwen'
        else:
            return 'other'

class HardwareDetector:
    """하드웨어 사양 감지 및 최적화 설정 추천 - 안전성 강화"""

    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """하드웨어 사양 감지 - 안전한 방식"""
        hardware_info = {
            'cuda_available': False,
            'cuda_device_count': 0,
            'total_memory': 0,
            'available_memory': 0,
            'cpu_count': 0,
            'cpu_cores': 0
        }

        try:
            import psutil
            memory = psutil.virtual_memory()
            hardware_info.update({
                'total_memory': memory.total // (1024**3),
                'available_memory': memory.available // (1024**3),
                'cpu_count': psutil.cpu_count(),
                'cpu_cores': psutil.cpu_count(logical=False)
            })
        except Exception as e:
            logging.warning(f"Failed to get system memory info: {e}")

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
                    except Exception as e:
                        logging.warning(f"Failed to get GPU {i} info: {e}")
        except ImportError:
            logging.warning("PyTorch not available for CUDA detection")
        except Exception as e:
            logging.warning(f"Failed to detect CUDA: {e}")

        return hardware_info

    @staticmethod
    def recommend_config(model_size: str) -> ModelConfig:
        """모델 크기에 따른 안전한 최적 설정 추천"""
        hardware = HardwareDetector.detect_hardware()

        # 안전한 기본 설정
        config = ModelConfig(
            name="recommended",
            model_path="",
            device="auto",
            dtype="float16",
            load_in_4bit=True,  # 기본적으로 양자화 활성화
            trust_remote_code=False
        )

        # GPU 메모리에 따른 설정 조정
        if hardware['cuda_available']:
            total_gpu_memory = sum(
                hardware.get(f'gpu_{i}_memory', 0)
                for i in range(hardware['cuda_device_count'])
            )

            if model_size in ['7b', '8b']:
                if total_gpu_memory >= 24:
                    config.dtype = "float16"
                    config.load_in_4bit = False
                elif total_gpu_memory >= 16:
                    config.dtype = "float16"
                    config.load_in_4bit = True
                else:
                    config.dtype = "float16"
                    config.load_in_4bit = True
                    config.max_memory = "8GB"

            elif model_size in ['13b', '14b']:
                if total_gpu_memory >= 48:
                    config.dtype = "float16"
                    config.load_in_4bit = False
                elif total_gpu_memory >= 24:
                    config.dtype = "float16"
                    config.load_in_4bit = True
                else:
                    config.dtype = "float16"
                    config.load_in_4bit = True
                    config.cpu_offload = True

            elif model_size in ['70b', '72b']:
                if total_gpu_memory >= 160:
                    config.tensor_parallel_size = min(4, hardware['cuda_device_count'])
                    config.load_in_4bit = True  # 대용량 모델은 항상 양자화
                else:
                    # 대용량 모델은 권장하지 않음
                    logging.warning(f"Insufficient GPU memory for {model_size} model")
                    config.cpu_offload = True
                    config.load_in_4bit = True
        else:
            # CPU 전용
            config.device = "cpu"
            config.dtype = "float32"
            config.load_in_8bit = False  # CPU에서는 양자화 비활성화
            config.load_in_4bit = False

        return config

    @staticmethod
    def get_safe_batch_size(model_size: str, available_memory_gb: float) -> int:
        """안전한 배치 크기 추천"""
        if model_size in ['7b', '8b']:
            if available_memory_gb >= 16:
                return 8
            elif available_memory_gb >= 8:
                return 4
            else:
                return 1
        elif model_size in ['13b', '14b']:
            if available_memory_gb >= 24:
                return 4
            elif available_memory_gb >= 16:
                return 2
            else:
                return 1
        else:  # 대용량 모델
            return 1

class SafetyChecker:
    """안전성 검사기"""

    @staticmethod
    def check_memory_safety(config: ModelConfig) -> List[str]:
        """메모리 안전성 검사"""
        warnings = []

        try:
            import torch
            if torch.cuda.is_available() and config.device == "cuda":
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

                # 메모리 사용량 추정
                estimated_usage = SafetyChecker._estimate_memory_usage(config)

                if estimated_usage > total_memory * 0.9:
                    warnings.append(f"Estimated memory usage ({estimated_usage:.1f}GB) exceeds 90% of available GPU memory ({total_memory:.1f}GB)")
                    warnings.append("Consider enabling quantization or reducing model size")

        except Exception:
            pass

        return warnings

    @staticmethod
    def _estimate_memory_usage(config: ModelConfig) -> float:
        """메모리 사용량 추정"""
        # 간단한 추정 (실제로는 더 복잡)
        model_size_map = {
            '7b': 14,   # GB (float16)
            '13b': 26,
            '30b': 60,
            '70b': 140
        }

        # 모델 크기 추정
        for size, memory in model_size_map.items():
            if size in config.model_path.lower():
                if config.load_in_4bit:
                    return memory * 0.5
                elif config.load_in_8bit:
                    return memory * 0.75
                else:
                    return memory

        return 14  # 기본값

# ConfigManager의 별칭 (하위 호환성)
ConfigManager = RobustConfigManager

# 사용 예시
if __name__ == "__main__":
    print("=== 안전한 설정 관리 시스템 테스트 ===")

    # 설정 매니저 생성
    config_manager = RobustConfigManager()

    # 하드웨어 정보 출력
    hardware_info = HardwareDetector.detect_hardware()
    print("하드웨어 정보:")
    for key, value in hardware_info.items():
        print(f"  {key}: {value}")

    # 모델 설정 출력
    print("\n모델 설정:")
    for name, config in config_manager.model_configs.items():
        print(f"  {name}:")
        print(f"    경로: {config.model_path}")
        print(f"    타입: {config.model_type}")
        print(f"    양자화: 4bit={config.load_in_4bit}, 8bit={config.load_in_8bit}")

    # 안전성 검사
    print("\n안전성 검사:")
    for name, config in config_manager.model_configs.items():
        safety_warnings = SafetyChecker.check_memory_safety(config)
        if safety_warnings:
            print(f"  {name} 경고:")
            for warning in safety_warnings:
                print(f"    - {warning}")
        else:
            print(f"  {name}: 안전함")

    # 추천 설정
    print("\n7B 모델 추천 설정:")
    recommended = HardwareDetector.recommend_config("7b")
    print(f"  Device: {recommended.device}")
    print(f"  Dtype: {recommended.dtype}")
    print(f"  4-bit: {recommended.load_in_4bit}")
    print(f"  8-bit: {recommended.load_in_8bit}")

    # 안전한 배치 크기
    available_memory = hardware_info.get('gpu_0_memory', 8)
    safe_batch = HardwareDetector.get_safe_batch_size("7b", available_memory)
    print(f"  권장 배치 크기: {safe_batch}")

    print("\n✅ 설정 관리 시스템 테스트 완료")


@dataclass
class AdvancedOptimizationConfig:
    """고급 최적화 설정"""
    enable_streaming: bool = False
    enable_ab_testing: bool = False
    cloud_deployment: bool = False
    web_ui_enabled: bool = False

    # vLLM 고급 설정
    speculative_decoding: bool = False
    prefix_caching: bool = False
    chunked_prefill: bool = False

    # 모니터링 설정
    prometheus_metrics: bool = False
    grafana_dashboard: bool = False

    # 자동 스케일링
    auto_scaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 10


class CloudConfig:
    """클라우드 배포 설정"""

    def __init__(self):
        self.provider = "local"  # local, aws, gcp, azure
        self.instance_type = "auto"
        self.region = "auto"
        self.auto_shutdown = True
        self.cost_limit_usd = 10.0


class WebUIConfig:
    """웹 UI 설정"""

    def __init__(self):
        self.enabled = False
        self.host = "localhost"
        self.port = 8080
        self.auth_required = True
        self.ssl_enabled = False