"""
모델 설정 모듈
Important 문제 해결: 설정 파일 분리 - 모델 관련 설정
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import json

from .base_config import BaseConfig, ValidationResult, ConfigValidator


class ModelType(Enum):
    """모델 타입"""
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    OLLAMA = "ollama"
    TGI = "tgi"


class DataType(Enum):
    """데이터 타입"""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"


class DeviceType(Enum):
    """장치 타입"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class QuantizationConfig:
    """양자화 설정"""
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"

    def validate(self) -> ValidationResult:
        """양자화 설정 검증"""
        result = ValidationResult(True)

        if self.load_in_4bit and self.load_in_8bit:
            result.add_error("4-bit와 8-bit 양자화를 동시에 사용할 수 없습니다")

        valid_compute_dtypes = ["float16", "float32", "bfloat16"]
        if self.bnb_4bit_compute_dtype not in valid_compute_dtypes:
            result.add_error(f"지원하지 않는 compute dtype: {self.bnb_4bit_compute_dtype}")

        valid_quant_types = ["fp4", "nf4"]
        if self.bnb_4bit_quant_type not in valid_quant_types:
            result.add_error(f"지원하지 않는 양자화 타입: {self.bnb_4bit_quant_type}")

        return result


@dataclass
class VLLMConfig:
    """vLLM 특화 설정"""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.8
    swap_space: int = 4  # GB
    disable_log_stats: bool = True

    def validate(self) -> ValidationResult:
        """vLLM 설정 검증"""
        result = ValidationResult(True)

        if self.tensor_parallel_size < 1:
            result.add_error("tensor_parallel_size는 1 이상이어야 합니다")

        if self.pipeline_parallel_size < 1:
            result.add_error("pipeline_parallel_size는 1 이상이어야 합니다")

        if not (0.1 <= self.gpu_memory_utilization <= 0.95):
            result.add_error("gpu_memory_utilization은 0.1-0.95 범위여야 합니다")

        if self.max_model_len and self.max_model_len < 1:
            result.add_error("max_model_len은 1 이상이어야 합니다")

        if self.swap_space < 0:
            result.add_error("swap_space는 0 이상이어야 합니다")

        return result


@dataclass
class OllamaConfig:
    """Ollama 특화 설정"""
    base_url: str = "http://localhost:11434"
    timeout: float = 60.0
    keep_alive: Optional[str] = None

    def validate(self) -> ValidationResult:
        """Ollama 설정 검증"""
        result = ValidationResult(True)

        if not self.base_url.startswith(("http://", "https://")):
            result.add_error("base_url은 http:// 또는 https://로 시작해야 합니다")

        if self.timeout <= 0:
            result.add_error("timeout은 0보다 커야 합니다")

        return result


@dataclass
class ModelConfig(BaseConfig):
    """통합 모델 설정"""
    model_path: str = ""
    model_type: ModelType = ModelType.TRANSFORMERS
    device: DeviceType = DeviceType.AUTO
    dtype: DataType = DataType.FLOAT16

    # 보안 설정
    trust_remote_code: bool = False

    # 최적화 설정
    use_flash_attention: bool = False
    use_bettertransformer: bool = False
    torch_compile: bool = False

    # 메모리 설정
    max_memory: Optional[str] = None
    low_cpu_mem_usage: bool = True

    # 양자화 설정
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # 엔진별 특화 설정
    vllm_config: Optional[VLLMConfig] = None
    ollama_config: Optional[OllamaConfig] = None

    # 모델 메타데이터
    description: str = ""
    tags: List[str] = field(default_factory=list)
    recommended_use: str = ""
    hardware_requirements: str = ""

    def validate(self) -> bool:
        """모델 설정 검증"""
        result = ValidationResult(True)

        # 기본 필드 검증
        if not self.model_path:
            result.add_error("model_path가 비어있습니다")

        # 모델 타입별 호환성 검증
        if self.model_type == ModelType.VLLM:
            if self.device == DeviceType.CPU:
                result.add_error("vLLM은 CPU를 지원하지 않습니다")

            if not self.vllm_config:
                result.add_warning("vLLM 사용 시 vllm_config 설정을 권장합니다")
                self.vllm_config = VLLMConfig()

            vllm_result = self.vllm_config.validate()
            result.errors.extend(vllm_result.errors)
            result.warnings.extend(vllm_result.warnings)

        elif self.model_type == ModelType.OLLAMA:
            if not self.ollama_config:
                self.ollama_config = OllamaConfig()

            ollama_result = self.ollama_config.validate()
            result.errors.extend(ollama_result.errors)
            result.warnings.extend(ollama_result.warnings)

        # 장치별 호환성 검증
        if self.device == DeviceType.MPS:
            if self.quantization.load_in_4bit or self.quantization.load_in_8bit:
                result.add_error("MPS는 양자화를 지원하지 않습니다")

        # 양자화 설정 검증
        quant_result = self.quantization.validate()
        result.errors.extend(quant_result.errors)
        result.warnings.extend(quant_result.warnings)

        # 보안 관련 경고
        if self.trust_remote_code:
            result.add_warning("trust_remote_code=True는 보안 위험이 있습니다")

        # 성능 관련 권고사항
        if self.use_flash_attention and self.dtype != DataType.FLOAT16:
            result.add_warning("Flash Attention은 float16에서 최적화됩니다")

        if self.torch_compile and not self._check_torch_compile_support():
            result.add_warning("torch.compile은 PyTorch 2.0+ 및 Python 3.11+에서 지원됩니다")

        # 결과 저장
        if result.errors:
            self.metadata['validation_errors'] = result.errors
        if result.warnings:
            self.metadata['validation_warnings'] = result.warnings

        return len(result.errors) == 0

    def _check_torch_compile_support(self) -> bool:
        """torch.compile 지원 여부 확인"""
        try:
            import sys
            import torch

            # Python 3.11+ 및 PyTorch 2.0+ 확인
            python_ok = sys.version_info >= (3, 11)
            torch_ok = hasattr(torch, 'compile')

            return python_ok and torch_ok
        except:
            return False

    def get_memory_estimate(self) -> float:
        """메모리 사용량 추정 (GB)"""
        # 모델 크기 기반 간단한 추정
        size_patterns = {
            '0.5b': 1.0, '1b': 2.0, '1.5b': 3.0,
            '3b': 6.0, '7b': 14.0, '8b': 16.0,
            '13b': 26.0, '14b': 28.0,
            '30b': 60.0, '32b': 64.0,
            '70b': 140.0, '72b': 144.0
        }

        base_memory = 14.0  # 기본값 (7B 모델)
        path_lower = self.model_path.lower()

        for pattern, memory in size_patterns.items():
            if pattern in path_lower:
                base_memory = memory
                break

        # 양자화 적용
        if self.quantization.load_in_4bit:
            return base_memory * 0.5
        elif self.quantization.load_in_8bit:
            return base_memory * 0.75
        else:
            return base_memory

    def get_recommended_hardware(self) -> Dict[str, Any]:
        """권장 하드웨어 사양"""
        memory_gb = self.get_memory_estimate()

        recommendations = {
            'min_vram_gb': memory_gb,
            'recommended_vram_gb': memory_gb * 1.2,
            'min_ram_gb': 16,
            'recommended_ram_gb': 32,
            'gpu_compute_capability': '7.0+'
        }

        # vLLM 사용시 추가 요구사항
        if self.model_type == ModelType.VLLM and self.vllm_config:
            recommendations['min_vram_gb'] *= self.vllm_config.tensor_parallel_size
            recommendations['recommended_vram_gb'] *= self.vllm_config.tensor_parallel_size

        return recommendations

    def optimize_for_hardware(self, available_vram_gb: float,
                              device_count: int = 1) -> 'ModelConfig':
        """하드웨어에 맞게 설정 최적화"""
        optimized = ModelConfig(**self.to_dict())

        estimated_memory = self.get_memory_estimate()

        # 메모리가 부족한 경우 양자화 적용
        if estimated_memory > available_vram_gb * 0.9:
            if not optimized.quantization.load_in_4bit:
                optimized.quantization.load_in_4bit = True
                optimized.quantization.load_in_8bit = False
                estimated_memory *= 0.5

        # 여전히 부족한 경우 추가 최적화
        if estimated_memory > available_vram_gb * 0.9:
            optimized.dtype = DataType.FLOAT16
            if self.model_type == ModelType.VLLM and device_count > 1:
                if not optimized.vllm_config:
                    optimized.vllm_config = VLLMConfig()
                optimized.vllm_config.tensor_parallel_size = min(device_count, 4)

        return optimized

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (enum 처리 포함)"""
        data = super().to_dict()

        # Enum을 문자열로 변환
        data['model_type'] = self.model_type.value
        data['device'] = self.device.value
        data['dtype'] = self.dtype.value

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """딕셔너리에서 생성 (enum 처리 포함)"""
        # Enum 변환
        if 'model_type' in data and isinstance(data['model_type'], str):
            data['model_type'] = ModelType(data['model_type'])

        if 'device' in data and isinstance(data['device'], str):
            data['device'] = DeviceType(data['device'])

        if 'dtype' in data and isinstance(data['dtype'], str):
            data['dtype'] = DataType(data['dtype'])

        # 양자화 설정 처리
        if 'quantization' in data and isinstance(data['quantization'], dict):
            data['quantization'] = QuantizationConfig(**data['quantization'])

        # vLLM 설정 처리
        if 'vllm_config' in data and isinstance(data['vllm_config'], dict):
            data['vllm_config'] = VLLMConfig(**data['vllm_config'])

        # Ollama 설정 처리
        if 'ollama_config' in data and isinstance(data['ollama_config'], dict):
            data['ollama_config'] = OllamaConfig(**data['ollama_config'])

        return super().from_dict(data)


class ModelConfigManager:
    """모델 설정 관리자"""

    def __init__(self):
        self.configs: Dict[str, ModelConfig] = {}

    def add_config(self, name: str, config: ModelConfig):
        """설정 추가"""
        if not config.validate():
            raise ValueError(f"모델 설정 {name}이 유효하지 않습니다")

        self.configs[name] = config

    def get_config(self, name: str) -> Optional[ModelConfig]:
        """설정 조회"""
        return self.configs.get(name)

    def list_configs(self) -> List[str]:
        """설정 목록"""
        return list(self.configs.keys())

    def save_to_file(self, filepath: str):
        """파일로 저장"""
        data = {
            name: config.to_dict()
            for name, config in self.configs.items()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, filepath: str):
        """파일에서 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.configs = {}
        for name, config_data in data.items():
            try:
                config = ModelConfig.from_dict(config_data)
                if config.validate():
                    self.configs[name] = config
                else:
                    print(f"⚠️ 모델 설정 {name} 검증 실패, 건너뜀")
            except Exception as e:
                print(f"⚠️ 모델 설정 {name} 로드 실패: {e}")

    def create_default_configs(self) -> Dict[str, ModelConfig]:
        """기본 설정들 생성"""
        default_configs = {}

        # Qwen 2.5 시리즈
        qwen_sizes = ["0.5b", "1.5b", "3b", "7b", "14b", "32b", "72b"]
        for size in qwen_sizes:
            name = f"qwen2.5-{size}"
            config = ModelConfig(
                name=name,
                model_path=f"Qwen/Qwen2.5-{size.upper()}-Instruct",
                model_type=ModelType.TRANSFORMERS,
                device=DeviceType.AUTO,
                dtype=DataType.FLOAT16,
                quantization=QuantizationConfig(load_in_4bit=size in ["14b", "32b", "72b"]),
                description=f"Qwen 2.5 {size.upper()} Instruct model",
                tags=["qwen", "instruct", "multilingual"],
                recommended_use="Korean NLP, reasoning, coding"
            )
            default_configs[name] = config

        # Llama 3 시리즈
        llama_configs = [
            ("llama3-8b", "meta-llama/Meta-Llama-3-8B-Instruct"),
            ("llama3-70b", "meta-llama/Meta-Llama-3-70B-Instruct"),
            ("llama3.1-8b", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
            ("llama3.1-70b", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
        ]

        for name, path in llama_configs:
            is_large = "70b" in name
            config = ModelConfig(
                name=name,
                model_path=path,
                model_type=ModelType.VLLM if is_large else ModelType.TRANSFORMERS,
                device=DeviceType.AUTO,
                dtype=DataType.FLOAT16,
                quantization=QuantizationConfig(load_in_4bit=is_large),
                vllm_config=VLLMConfig(tensor_parallel_size=4) if is_large else None,
                use_flash_attention=True,
                description=f"Meta Llama {name.replace('-', ' ').title()}",
                tags=["llama", "meta", "instruct"],
                recommended_use="General purpose, reasoning, coding"
            )
            default_configs[name] = config

        return default_configs


# 사용 예시
if __name__ == "__main__":
    print("=== 모델 설정 시스템 테스트 ===")

    # 모델 설정 생성
    config = ModelConfig(
        name="test-qwen-7b",
        model_path="Qwen/Qwen2.5-7B-Instruct",
        model_type=ModelType.TRANSFORMERS,
        device=DeviceType.AUTO,
        dtype=DataType.FLOAT16,
        quantization=QuantizationConfig(load_in_4bit=True),
        description="Test Qwen 7B model"
    )

    # 검증
    is_valid = config.validate()
    print(f"설정 유효성: {is_valid}")

    if 'validation_warnings' in config.metadata:
        print(f"경고: {config.metadata['validation_warnings']}")

    # 메모리 추정
    memory_gb = config.get_memory_estimate()
    print(f"예상 메모리 사용량: {memory_gb:.1f}GB")

    # 권장 하드웨어
    hardware = config.get_recommended_hardware()
    print(f"권장 하드웨어: {hardware}")

    # 하드웨어 최적화
    optimized = config.optimize_for_hardware(8.0, 1)
    print(f"최적화된 설정: 4bit={optimized.quantization.load_in_4bit}")

    # 설정 관리자 테스트
    manager = ModelConfigManager()
    manager.add_config("test", config)

    # 기본 설정들 생성
    defaults = manager.create_default_configs()
    print(f"기본 설정 수: {len(defaults)}")

    for name in list(defaults.keys())[:3]:  # 처음 3개만 출력
        print(f"  {name}: {defaults[name].model_path}")

    print("\n✅ 모델 설정 시스템 테스트 완료")