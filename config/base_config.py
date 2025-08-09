"""
설정 분리 - 기본 설정 모듈
Important 문제 해결: 설정 파일 분리
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from abc import ABC, abstractmethod


@dataclass
class BaseConfig(ABC):
    """기본 설정 클래스"""
    name: str
    version: str = "2.0.0"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """초기화 후 검증"""
        self.validate()

    @abstractmethod
    def validate(self) -> bool:
        """설정 검증"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """딕셔너리에서 생성"""
        # 클래스에 없는 필드 제거
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str):
        """오류 추가"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """경고 추가"""
        self.warnings.append(message)

    def has_issues(self) -> bool:
        """문제가 있는지 확인"""
        return len(self.errors) > 0 or len(self.warnings) > 0


class ConfigValidator:
    """설정 검증기"""

    @staticmethod
    def validate_file_path(path: str, must_exist: bool = True) -> ValidationResult:
        """파일 경로 검증"""
        result = ValidationResult(True)

        if not path:
            result.add_error("경로가 비어있습니다")
            return result

        path_obj = Path(path)

        if must_exist and not path_obj.exists():
            result.add_error(f"파일이 존재하지 않습니다: {path}")

        if must_exist and path_obj.exists() and not path_obj.is_file():
            result.add_error(f"파일이 아닙니다: {path}")

        # 권한 확인
        if path_obj.exists():
            if not os.access(path, os.R_OK):
                result.add_error(f"읽기 권한이 없습니다: {path}")

        return result

    @staticmethod
    def validate_memory_size(size_str: str) -> ValidationResult:
        """메모리 크기 검증 (예: "8GB", "512MB")"""
        result = ValidationResult(True)

        if not size_str:
            result.add_error("메모리 크기가 비어있습니다")
            return result

        size_str = size_str.upper().strip()

        # 패턴 확인
        import re
        pattern = r'^(\d+(?:\.\d+)?)(GB|MB|TB)$'
        match = re.match(pattern, size_str)

        if not match:
            result.add_error(f"잘못된 메모리 크기 형식: {size_str} (예: 8GB, 512MB)")
            return result

        value, unit = match.groups()
        value = float(value)

        # 합리적인 범위 확인
        if unit == "MB" and (value < 1 or value > 1024 * 1024):
            result.add_warning(f"비정상적인 메모리 크기: {size_str}")
        elif unit == "GB" and (value < 0.001 or value > 1024):
            result.add_warning(f"비정상적인 메모리 크기: {size_str}")
        elif unit == "TB" and (value < 0.001 or value > 100):
            result.add_warning(f"비정상적인 메모리 크기: {size_str}")

        return result

    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float,
                      name: str) -> ValidationResult:
        """범위 검증"""
        result = ValidationResult(True)

        if value < min_val:
            result.add_error(f"{name}이 최소값보다 작습니다: {value} < {min_val}")
        elif value > max_val:
            result.add_error(f"{name}이 최대값보다 큽니다: {value} > {max_val}")

        return result

    @staticmethod
    def validate_choices(value: str, choices: List[str], name: str) -> ValidationResult:
        """선택지 검증"""
        result = ValidationResult(True)

        if value not in choices:
            result.add_error(f"{name}의 값이 유효하지 않습니다: {value}, 선택 가능: {choices}")

        return result


class EnvironmentManager:
    """환경 변수 관리"""

    # 안전한 기본값들
    SAFE_DEFAULTS = {
        'TOKENIZERS_PARALLELISM': 'false',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'HF_HOME': str(Path.home() / '.cache' / 'huggingface'),
        'TRANSFORMERS_CACHE': str(Path.home() / '.cache' / 'huggingface' / 'transformers'),
        'OMP_NUM_THREADS': '4',
        'PYTHONPATH': '.',
    }

    @classmethod
    def setup_safe_environment(cls):
        """안전한 환경 변수 설정"""
        for key, value in cls.SAFE_DEFAULTS.items():
            if key not in os.environ:
                os.environ[key] = value
                logging.info(f"환경 변수 설정: {key}={value}")

    @classmethod
    def get_env_info(cls) -> Dict[str, str]:
        """환경 정보 반환"""
        return {key: os.environ.get(key, "Not Set") for key in cls.SAFE_DEFAULTS.keys()}


class PathManager:
    """경로 관리"""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

        # 기본 디렉토리들
        self.config_dir = self.base_dir / "config"
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "optimization_results"
        self.logs_dir = self.base_dir / "logs"
        self.cache_dir = self.base_dir / ".cache"

        self._create_directories()

    def _create_directories(self):
        """필요한 디렉토리 생성"""
        for directory in [self.config_dir, self.data_dir, self.results_dir,
                         self.logs_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_config_path(self, filename: str) -> Path:
        """설정 파일 경로"""
        return self.config_dir / filename

    def get_data_path(self, filename: str) -> Path:
        """데이터 파일 경로"""
        return self.data_dir / filename

    def get_results_path(self, filename: str) -> Path:
        """결과 파일 경로"""
        return self.results_dir / filename

    def get_logs_path(self, filename: str) -> Path:
        """로그 파일 경로"""
        return self.logs_dir / filename

    def get_cache_path(self, filename: str) -> Path:
        """캐시 파일 경로"""
        return self.cache_dir / filename


class LoggingConfig:
    """로깅 설정"""

    @staticmethod
    def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
        """로깅 설정"""
        # 레벨 변환
        numeric_level = getattr(logging, level.upper(), logging.INFO)

        # 포매터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # 파일 핸들러 (선택적)
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # 외부 라이브러리 로깅 레벨 조정
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

        logging.info(f"로깅 설정 완료: 레벨={level}, 파일={log_file}")


@dataclass
class SystemRequirements:
    """시스템 요구사항"""
    min_python_version: str = "3.8.0"
    min_memory_gb: float = 8.0
    min_disk_space_gb: float = 20.0
    required_packages: List[str] = field(default_factory=lambda: [
        "torch", "transformers", "numpy", "pandas", "optuna"
    ])

    def check_requirements(self) -> ValidationResult:
        """시스템 요구사항 확인"""
        result = ValidationResult(True)

        # Python 버전 확인
        import sys
        current_version = ".".join(map(str, sys.version_info[:3]))
        if self._version_compare(current_version, self.min_python_version) < 0:
            result.add_error(f"Python 버전 부족: {current_version} < {self.min_python_version}")

        # 메모리 확인
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < self.min_memory_gb:
                result.add_warning(f"메모리 부족: {memory_gb:.1f}GB < {self.min_memory_gb}GB")
        except ImportError:
            result.add_warning("psutil을 설치하여 메모리 확인을 권장합니다")

        # 디스크 공간 확인
        try:
            import shutil
            disk_space_gb = shutil.disk_usage(".").free / (1024**3)
            if disk_space_gb < self.min_disk_space_gb:
                result.add_warning(f"디스크 공간 부족: {disk_space_gb:.1f}GB < {self.min_disk_space_gb}GB")
        except:
            result.add_warning("디스크 공간 확인 실패")

        # 패키지 확인
        missing_packages = []
        for package in self.required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            result.add_error(f"누락된 필수 패키지: {missing_packages}")

        return result

    def _version_compare(self, version1: str, version2: str) -> int:
        """버전 비교 (-1: v1 < v2, 0: v1 == v2, 1: v1 > v2)"""
        v1_parts = [int(x) for x in version1.split('.')]
        v2_parts = [int(x) for x in version2.split('.')]

        # 길이 맞추기
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))

        for v1, v2 in zip(v1_parts, v2_parts):
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1

        return 0


class ConfigRegistry:
    """설정 레지스트리"""

    def __init__(self):
        self._configs: Dict[str, BaseConfig] = {}
        self._validators: Dict[str, Callable] = {}

    def register_config(self, name: str, config: BaseConfig):
        """설정 등록"""
        self._configs[name] = config

    def get_config(self, name: str) -> Optional[BaseConfig]:
        """설정 조회"""
        return self._configs.get(name)

    def list_configs(self) -> List[str]:
        """설정 목록"""
        return list(self._configs.keys())

    def validate_all(self) -> Dict[str, ValidationResult]:
        """모든 설정 검증"""
        results = {}
        for name, config in self._configs.items():
            try:
                results[name] = ValidationResult(config.validate())
            except Exception as e:
                result = ValidationResult(False)
                result.add_error(f"검증 중 오류: {e}")
                results[name] = result
        return results


# 전역 설정 레지스트리
_global_config_registry = ConfigRegistry()


def get_config_registry() -> ConfigRegistry:
    """전역 설정 레지스트리 반환"""
    return _global_config_registry


# 사용 예시
if __name__ == "__main__":
    print("=== 기본 설정 시스템 테스트 ===")

    # 환경 설정
    EnvironmentManager.setup_safe_environment()
    env_info = EnvironmentManager.get_env_info()
    print("환경 변수:")
    for key, value in env_info.items():
        print(f"  {key}: {value}")

    # 경로 관리
    path_manager = PathManager()
    print(f"\n경로 설정:")
    print(f"  설정: {path_manager.config_dir}")
    print(f"  데이터: {path_manager.data_dir}")
    print(f"  결과: {path_manager.results_dir}")
    print(f"  로그: {path_manager.logs_dir}")

    # 로깅 설정
    LoggingConfig.setup_logging("INFO", path_manager.get_logs_path("test.log"))

    # 시스템 요구사항 확인
    requirements = SystemRequirements()
    req_result = requirements.check_requirements()
    print(f"\n시스템 요구사항:")
    print(f"  유효성: {req_result.is_valid}")
    if req_result.errors:
        print(f"  오류: {req_result.errors}")
    if req_result.warnings:
        print(f"  경고: {req_result.warnings}")

    # 검증 테스트
    validator = ConfigValidator()

    # 파일 경로 검증
    path_result = validator.validate_file_path("test.txt", must_exist=False)
    print(f"\n파일 경로 검증: {path_result.is_valid}")

    # 메모리 크기 검증
    memory_result = validator.validate_memory_size("8GB")
    print(f"메모리 크기 검증: {memory_result.is_valid}")

    # 범위 검증
    range_result = validator.validate_range(0.5, 0.0, 1.0, "temperature")
    print(f"범위 검증: {range_result.is_valid}")

    # 선택지 검증
    choice_result = validator.validate_choices("float16", ["float16", "float32"], "dtype")
    print(f"선택지 검증: {choice_result.is_valid}")

    print("\n✅ 기본 설정 시스템 테스트 완료")