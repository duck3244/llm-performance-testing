"""
안전성 강화된 오픈소스 LLM 모델 인터페이스
메모리 누수, 스레드 안전성, 성능 문제 해결
"""
import asyncio
import time
import torch
import psutil
import gc
import threading
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, AsyncIterator
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import weakref

from config import ModelConfig, InferenceParams, OptimizationConfig, get_resource_manager

@dataclass
class ModelResponse:
    """모델 응답 데이터 클래스"""
    content: str
    model: str
    generation_stats: Dict[str, Any]
    latency: float
    timestamp: datetime
    params: InferenceParams
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class BatchResponse:
    """배치 응답 데이터 클래스"""
    responses: List[ModelResponse]
    total_tokens: int
    avg_tokens_per_second: float
    peak_memory_usage: float
    total_latency: float
    throughput: float
    success_count: int
    error_count: int
    metadata: Optional[Dict[str, Any]] = None

class SafeModelInterface(ABC):
    """안전한 기본 모델 인터페이스"""

    def __init__(self, config: ModelConfig, optimization_config: OptimizationConfig):
        self.config = config
        self.optimization_config = optimization_config
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{config.name}")

        # 모델 상태
        self.model = None
        self.tokenizer = None
        self.device = None
        self._is_loaded = False
        self._load_lock = threading.RLock()

        # 안전성 관련
        self._memory_monitor = MemoryMonitor()
        self._request_limiter = RequestLimiter(max_concurrent=4)
        self._health_checker = HealthChecker()

        # 리소스 매니저에 등록
        self.resource_manager = get_resource_manager()
        self.resource_manager.register_model(config.name, self)

        # 모델 초기화
        self._initialize_model()

    @abstractmethod
    def _load_model_impl(self):
        """실제 모델 로딩 구현 (서브클래스에서 구현)"""
        pass

    def _initialize_model(self):
        """안전한 모델 초기화"""
        with self._load_lock:
            if self._is_loaded:
                return

            try:
                # 메모리 체크
                if not self._memory_monitor.check_available_memory(self.config):
                    raise RuntimeError("Insufficient memory for model loading")

                # 헬스 체크
                if not self._health_checker.check_system_health():
                    raise RuntimeError("System health check failed")

                # 실제 모델 로딩
                self._load_model_impl()
                self._is_loaded = True

                self.logger.info(f"Model {self.config.name} loaded successfully")

            except Exception as e:
                self.logger.error(f"Failed to initialize model {self.config.name}: {e}")
                self._cleanup()
                raise

    async def generate_async(self, prompt: str, params: InferenceParams) -> ModelResponse:
        """안전한 비동기 텍스트 생성"""
        # 요청 제한 확인
        async with self._request_limiter:
            return await self._generate_with_safety_checks(prompt, params)

    async def _generate_with_safety_checks(self, prompt: str, params: InferenceParams) -> ModelResponse:
        """안전성 검사를 포함한 생성"""
        start_time = time.time()

        try:
            # 모델 로드 상태 확인
            if not self._is_loaded:
                self._initialize_model()

            # 메모리 체크
            self._memory_monitor.check_memory_before_generation()

            # 입력 검증
            validated_prompt = self._validate_input(prompt)
            validated_params = self._validate_params(params)

            # 실제 생성
            response = await self._generate_impl(validated_prompt, validated_params)

            # 후처리
            response.latency = time.time() - start_time
            response.timestamp = datetime.now()
            response.model = self.config.name

            return response

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return ModelResponse(
                content="",
                model=self.config.name,
                generation_stats={},
                latency=time.time() - start_time,
                timestamp=datetime.now(),
                params=params,
                error=str(e)
            )

    @abstractmethod
    async def _generate_impl(self, prompt: str, params: InferenceParams) -> ModelResponse:
        """실제 생성 구현 (서브클래스에서 구현)"""
        pass

    def _validate_input(self, prompt: str) -> str:
        """입력 검증"""
        if not prompt or not prompt.strip():
            raise ValueError("Empty prompt")

        # 길이 제한
        if len(prompt) > 100000:  # 100K 문자 제한
            self.logger.warning(f"Prompt too long ({len(prompt)} chars), truncating")
            prompt = prompt[:100000]

        return prompt.strip()

    def _validate_params(self, params: InferenceParams) -> InferenceParams:
        """파라미터 검증"""
        # 이미 __post_init__에서 검증되지만 추가 검증
        if params.max_new_tokens > 4096:
            self.logger.warning(f"max_new_tokens too large ({params.max_new_tokens}), limiting to 4096")
            params.max_new_tokens = 4096

        return params

    async def generate_batch_async(self, prompts: List[str], params: InferenceParams,
                                  max_concurrent: int = None) -> BatchResponse:
        """안전한 배치 생성"""
        if not prompts:
            return BatchResponse([], 0, 0, 0, 0, 0, 0, 0)

        max_concurrent = max_concurrent or min(len(prompts), self._request_limiter.max_concurrent)
        start_time = time.time()

        # 세마포어로 동시 실행 제한
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(prompt: str) -> ModelResponse:
            async with semaphore:
                return await self.generate_async(prompt, params)

        # 배치 처리
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 처리
        valid_responses = []
        error_count = 0

        for response in responses:
            if isinstance(response, Exception):
                error_count += 1
                self.logger.error(f"Batch generation error: {response}")
            elif isinstance(response, ModelResponse):
                if response.error:
                    error_count += 1
                else:
                    valid_responses.append(response)

        # 통계 계산
        total_latency = time.time() - start_time
        total_tokens = sum(r.generation_stats.get('total_tokens', 0) for r in valid_responses)
        avg_tokens_per_second = total_tokens / total_latency if total_latency > 0 else 0
        throughput = len(valid_responses) / total_latency if total_latency > 0 else 0
        peak_memory = self._memory_monitor.get_peak_memory_usage()

        return BatchResponse(
            responses=valid_responses,
            total_tokens=total_tokens,
            avg_tokens_per_second=avg_tokens_per_second,
            peak_memory_usage=peak_memory,
            total_latency=total_latency,
            throughput=throughput,
            success_count=len(valid_responses),
            error_count=error_count
        )

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        try:
            info = {
                'name': self.config.name,
                'path': self.config.model_path,
                'type': self.config.model_type,
                'device': str(self.device) if self.device else 'unknown',
                'loaded': self._is_loaded,
                'memory_usage_mb': self._get_memory_usage(),
                'health_status': self._health_checker.get_status()
            }

            if self._is_loaded and self.model:
                try:
                    param_count = sum(p.numel() for p in self.model.parameters())
                    info['parameter_count'] = param_count
                    info['parameter_count_human'] = self._format_parameter_count(param_count)
                except:
                    pass

            return info

        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {'name': self.config.name, 'error': str(e)}

    def _format_parameter_count(self, count: int) -> str:
        """파라미터 수를 읽기 쉽게 포맷"""
        if count >= 1e9:
            return f"{count/1e9:.1f}B"
        elif count >= 1e6:
            return f"{count/1e6:.1f}M"
        elif count >= 1e3:
            return f"{count/1e3:.1f}K"
        else:
            return str(count)

    def _get_memory_usage(self) -> float:
        """메모리 사용량 반환 (MB)"""
        return self._memory_monitor.get_current_memory_usage()

    def _cleanup(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # 모델을 CPU로 이동
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()

                # 메모리에서 제거
                del self.model
                self.model = None

            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # CUDA 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 가비지 컬렉션
            gc.collect()

            self._is_loaded = False
            self.logger.info(f"Model {self.config.name} cleaned up")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """소멸자"""
        self._cleanup()

class SafeTransformersInterface(SafeModelInterface):
    """안전한 Transformers 인터페이스"""

    def _load_model_impl(self):
        """Transformers 모델 로딩"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            self.logger.info(f"Loading Transformers model: {self.config.model_path}")

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="left",
                use_fast=True  # fast tokenizer 사용
            )

            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 양자화 설정
            quantization_config = None
            if self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # 장치 맵 설정
            device_map = self.config.device
            if device_map == "auto":
                device_map = "auto"
            elif device_map == "cpu":
                device_map = "cpu"
            else:
                device_map = {"": self.config.device}

            # 모델 로드 arguments
            model_kwargs = {
                'trust_remote_code': self.config.trust_remote_code,
                'torch_dtype': getattr(torch, self.config.dtype),
                'device_map': device_map,
                'low_cpu_mem_usage': True,  # 메모리 효율성
            }

            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config

            # Flash Attention 설정
            if self.config.use_flash_attention:
                try:
                    model_kwargs['attn_implementation'] = "flash_attention_2"
                except:
                    self.logger.warning("Flash Attention 2 not available, using default")

            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )

            # 최적화 적용
            self._apply_optimizations()

            # 장치 정보 저장
            self.device = next(self.model.parameters()).device
            self.logger.info(f"Model loaded on device: {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load Transformers model: {e}")
            raise

    def _apply_optimizations(self):
        """모델 최적화 적용"""
        try:
            # BetterTransformer (안전하게 적용)
            if self.optimization_config.enable_bettertransformer:
                try:
                    self.model = self.model.to_bettertransformer()
                    self.logger.info("BetterTransformer enabled")
                except Exception as e:
                    self.logger.warning(f"BetterTransformer failed: {e}")

            # Torch Compile (안전하게 적용)
            if self.optimization_config.enable_torch_compile:
                try:
                    # Python 3.11+ 및 적절한 PyTorch 버전에서만
                    import sys
                    if sys.version_info >= (3, 11):
                        self.model = torch.compile(self.model, mode="default")
                        self.logger.info("Torch compile enabled")
                    else:
                        self.logger.warning("Torch compile requires Python 3.11+")
                except Exception as e:
                    self.logger.warning(f"Torch compile failed: {e}")

            # Gradient checkpointing (메모리 절약)
            if self.optimization_config.gradient_checkpointing:
                try:
                    self.model.gradient_checkpointing_enable()
                    self.logger.info("Gradient checkpointing enabled")
                except Exception as e:
                    self.logger.warning(f"Gradient checkpointing failed: {e}")

        except Exception as e:
            self.logger.error(f"Optimization application failed: {e}")

    async def _generate_impl(self, prompt: str, params: InferenceParams) -> ModelResponse:
        """Transformers 생성 구현"""
        def generate_sync():
            # 토큰화
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096  # 안전한 최대 길이
            ).to(self.device)

            input_length = inputs['input_ids'].shape[1]

            # 생성 설정
            generation_kwargs = {
                'max_new_tokens': params.max_new_tokens,
                'min_new_tokens': params.min_new_tokens,
                'temperature': params.temperature,
                'top_p': params.top_p,
                'top_k': params.top_k,
                'repetition_penalty': params.repetition_penalty,
                'length_penalty': params.length_penalty,
                'do_sample': params.do_sample,
                'num_beams': params.num_beams,
                'early_stopping': params.early_stopping,
                'use_cache': params.use_cache,
                'pad_token_id': params.pad_token_id or self.tokenizer.pad_token_id,
                'eos_token_id': params.eos_token_id or self.tokenizer.eos_token_id,
                'num_return_sequences': params.num_return_sequences,
            }

            # 생성 실행
            generation_start = time.time()
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            generation_time = time.time() - generation_start

            # 결과 디코딩
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # 통계 계산
            total_tokens = len(generated_tokens)
            tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

            return ModelResponse(
                content=generated_text,
                model=self.config.name,
                generation_stats={
                    'input_tokens': input_length,
                    'output_tokens': total_tokens,
                    'total_tokens': input_length + total_tokens,
                    'tokens_per_second': tokens_per_second,
                    'generation_time': generation_time,
                    'memory_usage_mb': self._get_memory_usage()
                },
                latency=0,  # 나중에 설정됨
                timestamp=datetime.now(),
                params=params
            )

        # 스레드풀에서 동기 함수 실행
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, generate_sync)

class MemoryMonitor:
    """메모리 모니터링 클래스"""

    def __init__(self):
        self.peak_memory = 0
        self.lock = threading.Lock()

    def check_available_memory(self, config: ModelConfig) -> bool:
        """모델 로딩 전 메모리 체크"""
        try:
            if config.device == "cuda" and torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory
                # 간단한 메모리 추정
                estimated_usage = self._estimate_memory_usage(config)
                return estimated_usage < available_memory * 0.9
            return True
        except:
            return True

    def _estimate_memory_usage(self, config: ModelConfig) -> int:
        """메모리 사용량 추정 (바이트)"""
        # 모델 크기 추정 (매우 간단한 버전)
        size_estimates = {
            '7b': 14 * 1024**3,   # 14GB
            '13b': 26 * 1024**3,  # 26GB
            '30b': 60 * 1024**3,  # 60GB
            '70b': 140 * 1024**3, # 140GB
        }

        base_size = 14 * 1024**3  # 기본 7B 크기
        path_lower = config.model_path.lower()

        for size, memory in size_estimates.items():
            if size in path_lower:
                base_size = memory
                break

        # 양자화 적용
        if config.load_in_4bit:
            return int(base_size * 0.5)
        elif config.load_in_8bit:
            return int(base_size * 0.75)
        else:
            return base_size

    def check_memory_before_generation(self):
        """생성 전 메모리 체크"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory

                if allocated / total > 0.95:
                    torch.cuda.empty_cache()

                if reserved / total > 0.98:
                    raise RuntimeError("GPU memory usage too high")
        except Exception as e:
            logging.warning(f"Memory check failed: {e}")

    def get_current_memory_usage(self) -> float:
        """현재 메모리 사용량 (MB)"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**2
            else:
                process = psutil.Process()
                return process.memory_info().rss / 1024**2
        except:
            return 0.0

    def get_peak_memory_usage(self) -> float:
        """피크 메모리 사용량 업데이트 및 반환"""
        current = self.get_current_memory_usage()
        with self.lock:
            self.peak_memory = max(self.peak_memory, current)
            return self.peak_memory

class RequestLimiter:
    """요청 제한기"""

    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()

class HealthChecker:
    """시스템 헬스 체크"""

    def __init__(self):
        self.last_check = 0
        self.check_interval = 60  # 1분
        self.status = {}

    def check_system_health(self) -> bool:
        """시스템 헬스 체크"""
        current_time = time.time()

        # 캐시된 결과 사용 (1분 이내)
        if current_time - self.last_check < self.check_interval:
            return self.status.get('healthy', True)

        try:
            # CPU 사용률 체크
            cpu_percent = psutil.cpu_percent(interval=1)

            # 메모리 사용률 체크
            memory = psutil.virtual_memory()

            # 디스크 사용률 체크
            disk = psutil.disk_usage('/')

            # 건강성 판단
            healthy = (
                cpu_percent < 90 and
                memory.percent < 95 and
                disk.percent < 95
            )

            self.status = {
                'healthy': healthy,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'last_check': current_time
            }

            self.last_check = current_time
            return healthy

        except Exception as e:
            logging.warning(f"Health check failed: {e}")
            return True  # 체크 실패 시 통과로 처리

    def get_status(self) -> Dict[str, Any]:
        """헬스 상태 반환"""
        return self.status.copy()

class SafeModelManager:
    """안전한 모델 매니저"""

    def __init__(self, optimization_config: OptimizationConfig):
        self.optimization_config = optimization_config
        self.models: Dict[str, SafeModelInterface] = {}
        self.model_locks: Dict[str, threading.RLock] = {}
        self.logger = logging.getLogger(__name__)

        # 전역 락
        self._global_lock = threading.RLock()

    def register_model(self, name: str, config: ModelConfig, force_reload: bool = False):
        """안전한 모델 등록"""
        with self._global_lock:
            # 기존 모델 정리 (필요한 경우)
            if name in self.models and force_reload:
                self.unregister_model(name)

            if name in self.models:
                self.logger.info(f"Model {name} already registered")
                return

            # 모델별 락 생성
            self.model_locks[name] = threading.RLock()

            try:
                # 모델 타입에 따른 인터페이스 선택
                if config.model_type == "transformers":
                    interface = SafeTransformersInterface(config, self.optimization_config)
                else:
                    # 다른 타입들도 SafeTransformersInterface로 처리 (일단)
                    # 추후 vLLM, Ollama 등의 안전한 구현 추가
                    interface = SafeTransformersInterface(config, self.optimization_config)

                self.models[name] = interface
                self.logger.info(f"Model {name} registered successfully")

            except Exception as e:
                self.logger.error(f"Failed to register model {name}: {e}")
                # 실패 시 정리
                if name in self.model_locks:
                    del self.model_locks[name]
                raise

    def unregister_model(self, name: str):
        """모델 등록 해제"""
        with self._global_lock:
            if name not in self.models:
                return

            with self.model_locks.get(name, threading.RLock()):
                try:
                    # 모델 정리
                    model = self.models[name]
                    model._cleanup()

                    # 딕셔너리에서 제거
                    del self.models[name]
                    if name in self.model_locks:
                        del self.model_locks[name]

                    self.logger.info(f"Model {name} unregistered")

                except Exception as e:
                    self.logger.error(f"Error unregistering model {name}: {e}")

    def get_model(self, name: str) -> Optional[SafeModelInterface]:
        """안전한 모델 인터페이스 반환"""
        with self._global_lock:
            if name not in self.models:
                self.logger.warning(f"Model {name} not found")
                return None

            return self.models[name]

    def list_models(self) -> List[str]:
        """등록된 모델 목록"""
        with self._global_lock:
            return list(self.models.keys())

    def get_models_info(self) -> Dict[str, Dict[str, Any]]:
        """모든 모델 정보"""
        with self._global_lock:
            return {
                name: model.get_model_info()
                for name, model in self.models.items()
            }

    async def generate_batch_all_models(self, prompts: List[str],
                                       params: InferenceParams) -> Dict[str, BatchResponse]:
        """모든 모델에서 배치 생성 (비교용)"""
        results = {}

        for name, model in self.models.items():
            try:
                self.logger.info(f"Running batch generation on {name}")
                result = await model.generate_batch_async(prompts, params)
                results[name] = result
            except Exception as e:
                self.logger.error(f"Batch generation failed for {name}: {e}")
                results[name] = BatchResponse([], 0, 0, 0, 0, 0, 0, len(prompts))

        return results

    def cleanup_all(self):
        """모든 모델 정리"""
        with self._global_lock:
            model_names = list(self.models.keys())
            for name in model_names:
                self.unregister_model(name)

# 사용 예시
if __name__ == "__main__":
    import asyncio
    from config import ModelConfig, OptimizationConfig, InferenceParams

    async def test_safe_interface():
        print("=== 안전한 모델 인터페이스 테스트 ===")

        # 설정
        model_config = ModelConfig(
            name="test-model",
            model_path="microsoft/DialoGPT-medium",  # 작은 테스트 모델
            model_type="transformers",
            device="auto",
            dtype="float16",
            load_in_4bit=True,
            trust_remote_code=False
        )

        optimization_config = OptimizationConfig(
            enable_torch_compile=False,
            enable_bettertransformer=False,
            auto_cleanup=True
        )

        try:
            # 모델 매니저 생성
            manager = SafeModelManager(optimization_config)

            # 모델 등록
            manager.register_model("test", model_config)
            print("✅ 모델 등록 완료")

            # 모델 정보 확인
            model = manager.get_model("test")
            if model:
                info = model.get_model_info()
                print(f"📋 모델 정보: {info}")

                # 단일 생성 테스트
                params = InferenceParams(
                    temperature=0.7,
                    max_new_tokens=50,
                    top_p=0.9
                )

                response = await model.generate_async("안녕하세요!", params)
                print(f"🔤 생성 결과: {response.content}")
                print(f"⏱️ 지연시간: {response.latency:.2f}초")

                # 배치 생성 테스트
                prompts = ["안녕하세요!", "오늘 날씨는?", "AI에 대해 설명해주세요."]
                batch_response = await model.generate_batch_async(prompts, params)
                print(f"📦 배치 결과: {batch_response.success_count}개 성공, {batch_response.error_count}개 실패")
                print(f"🚀 처리 속도: {batch_response.avg_tokens_per_second:.1f} tokens/sec")

            print("✅ 모든 테스트 완료")

        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 정리
            if 'manager' in locals():
                manager.cleanup_all()

            # 전역 리소스 정리
            from config import cleanup_resources
            cleanup_resources()

    # 테스트 실행
    asyncio.run(test_safe_interface())