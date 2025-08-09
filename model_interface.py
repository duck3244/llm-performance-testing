"""
오픈소스 LLM 모델 인터페이스 및 최적화된 추론 엔진
"""
import time
import torch
import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

from config import ModelConfig, InferenceParams, OptimizationConfig

@dataclass
class ModelResponse:
    """모델 응답 데이터 클래스"""
    content: str
    model: str
    generation_stats: Dict[str, Any]  # tokens/sec, memory usage, etc.
    latency: float
    timestamp: datetime
    params: InferenceParams
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class BatchResponse:
    """배치 응답 데이터 클래스"""
    responses: List[ModelResponse]
    total_tokens: int
    avg_tokens_per_second: float
    peak_memory_usage: float
    total_latency: float
    throughput: float  # requests per second
    success_count: int
    error_count: int

@dataclass
class PerformanceMetrics:
    """성능 메트릭 클래스"""
    tokens_per_second: float
    memory_usage_mb: float
    gpu_utilization: float
    cpu_utilization: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float

class BaseModelInterface(ABC):
    """기본 모델 인터페이스"""

    def __init__(self, config: ModelConfig, optimization_config: OptimizationConfig):
        self.config = config
        self.optimization_config = optimization_config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.model = None
        self.tokenizer = None
        self.device = None
        self.performance_monitor = PerformanceMonitor()
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self):
        """모델 초기화"""
        pass

    @abstractmethod
    async def generate_async(self, prompt: str, params: InferenceParams) -> ModelResponse:
        """비동기 텍스트 생성"""
        pass

    def generate(self, prompt: str, params: InferenceParams) -> ModelResponse:
        """동기 텍스트 생성"""
        return asyncio.run(self.generate_async(prompt, params))

    async def generate_batch_async(self, prompts: List[str], params: InferenceParams,
                                  max_concurrent: int = 4) -> BatchResponse:
        """배치 비동기 생성"""
        start_time = time.time()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(prompt: str) -> Optional[ModelResponse]:
            async with semaphore:
                try:
                    return await self.generate_async(prompt, params)
                except Exception as e:
                    self.logger.error(f"Error generating response: {e}")
                    return None

        # 성능 모니터링 시작
        self.performance_monitor.start_monitoring()

        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)

        # 성능 모니터링 종료
        perf_metrics = self.performance_monitor.stop_monitoring()

        total_latency = time.time() - start_time
        valid_responses = [r for r in responses if r is not None]

        # 통계 계산
        total_tokens = sum(r.generation_stats.get('total_tokens', 0) for r in valid_responses)
        avg_tokens_per_second = total_tokens / total_latency if total_latency > 0 else 0
        throughput = len(valid_responses) / total_latency if total_latency > 0 else 0

        return BatchResponse(
            responses=valid_responses,
            total_tokens=total_tokens,
            avg_tokens_per_second=avg_tokens_per_second,
            peak_memory_usage=perf_metrics.memory_usage_mb,
            total_latency=total_latency,
            throughput=throughput,
            success_count=len(valid_responses),
            error_count=len(responses) - len(valid_responses)
        )

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        try:
            model_size = sum(p.numel() for p in self.model.parameters())
            return {
                'model_name': self.config.name,
                'model_path': self.config.model_path,
                'model_type': self.config.model_type,
                'device': str(self.device),
                'dtype': self.config.dtype,
                'parameter_count': model_size,
                'memory_footprint_mb': self._get_memory_usage(),
                'quantization': {
                    'load_in_4bit': self.config.load_in_4bit,
                    'load_in_8bit': self.config.load_in_8bit
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {}

    def _get_memory_usage(self) -> float:
        """메모리 사용량 반환 (MB)"""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024

class TransformersInterface(BaseModelInterface):
    """HuggingFace Transformers 인터페이스"""

    def _initialize_model(self):
        """Transformers 모델 초기화"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch

            self.logger.info(f"Loading model: {self.config.model_path}")

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="left"
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

            # 모델 로드
            model_kwargs = {
                'trust_remote_code': self.config.trust_remote_code,
                'torch_dtype': getattr(torch, self.config.dtype),
                'device_map': self.config.device if self.config.device != "auto" else "auto",
                'quantization_config': quantization_config,
            }

            if self.config.use_flash_attention:
                model_kwargs['attn_implementation'] = "flash_attention_2"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )

            # 최적화 적용
            if self.optimization_config.enable_bettertransformer:
                try:
                    self.model = self.model.to_bettertransformer()
                    self.logger.info("BetterTransformer enabled")
                except Exception as e:
                    self.logger.warning(f"BetterTransformer failed: {e}")

            if self.optimization_config.enable_torch_compile:
                try:
                    self.model = torch.compile(self.model)
                    self.logger.info("Torch compile enabled")
                except Exception as e:
                    self.logger.warning(f"Torch compile failed: {e}")

            self.device = next(self.model.parameters()).device
            self.logger.info(f"Model loaded on device: {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise

    async def generate_async(self, prompt: str, params: InferenceParams) -> ModelResponse:
        """Transformers를 통한 비동기 텍스트 생성"""
        start_time = time.time()

        def generate_sync():
            # 토큰화
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
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
            with torch.no_grad():
                generation_start = time.time()
                outputs = self.model.generate(**inputs, **generation_kwargs)
                generation_time = time.time() - generation_start

            # 결과 디코딩
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # 통계 계산
            total_tokens = len(generated_tokens)
            tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

            return generated_text, {
                'input_tokens': input_length,
                'output_tokens': total_tokens,
                'total_tokens': input_length + total_tokens,
                'tokens_per_second': tokens_per_second,
                'generation_time': generation_time,
                'memory_usage_mb': self._get_memory_usage()
            }

        # 별도 스레드에서 실행
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            generated_text, stats = await loop.run_in_executor(executor, generate_sync)

        latency = time.time() - start_time

        return ModelResponse(
            content=generated_text,
            model=self.config.name,
            generation_stats=stats,
            latency=latency,
            timestamp=datetime.now(),
            params=params,
            metadata={'backend': 'transformers'}
        )

class VLLMInterface(BaseModelInterface):
    """vLLM 인터페이스 (고성능 추론)"""

    def _initialize_model(self):
        """vLLM 모델 초기화"""
        try:
            from vllm import LLM, SamplingParams

            self.logger.info(f"Loading vLLM model: {self.config.model_path}")

            # vLLM 설정
            llm_kwargs = {
                'model': self.config.model_path,
                'tensor_parallel_size': self.config.tensor_parallel_size,
                'dtype': self.config.dtype,
                'gpu_memory_utilization': self.config.gpu_memory_utilization,
                'trust_remote_code': self.config.trust_remote_code,
            }

            if self.config.max_model_len:
                llm_kwargs['max_model_len'] = self.config.max_model_len

            self.model = LLM(**llm_kwargs)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.logger.info(f"vLLM model loaded successfully")

        except ImportError:
            self.logger.error("vLLM not installed. Install with: pip install vllm")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize vLLM model: {e}")
            raise

    async def generate_async(self, prompt: str, params: InferenceParams) -> ModelResponse:
        """vLLM을 통한 비동기 텍스트 생성"""
        from vllm import SamplingParams

        start_time = time.time()

        def generate_sync():
            sampling_params = SamplingParams(
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                max_tokens=params.max_new_tokens,
                repetition_penalty=params.repetition_penalty,
                length_penalty=params.length_penalty,
                use_beam_search=params.num_beams > 1,
                best_of=params.num_beams if params.num_beams > 1 else 1,
                n=params.num_return_sequences,
                stop=None,
            )

            generation_start = time.time()
            outputs = self.model.generate([prompt], sampling_params)
            generation_time = time.time() - generation_start

            output = outputs[0]
            generated_text = output.outputs[0].text

            # 통계 계산
            input_tokens = len(output.prompt_token_ids)
            output_tokens = len(output.outputs[0].token_ids)
            total_tokens = input_tokens + output_tokens
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0

            return generated_text, {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'tokens_per_second': tokens_per_second,
                'generation_time': generation_time,
                'memory_usage_mb': self._get_memory_usage(),
                'finish_reason': output.outputs[0].finish_reason
            }

        # 별도 스레드에서 실행
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            generated_text, stats = await loop.run_in_executor(executor, generate_sync)

        latency = time.time() - start_time

        return ModelResponse(
            content=generated_text,
            model=self.config.name,
            generation_stats=stats,
            latency=latency,
            timestamp=datetime.now(),
            params=params,
            metadata={'backend': 'vllm'}
        )

class OllamaInterface(BaseModelInterface):
    """Ollama 인터페이스"""

    def _initialize_model(self):
        """Ollama 클라이언트 초기화"""
        try:
            import ollama

            self.client = ollama.Client(host=self.config.base_url)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 모델 존재 확인
            try:
                self.client.show(self.config.model_path)
                self.logger.info(f"Ollama model {self.config.model_path} is available")
            except:
                self.logger.info(f"Pulling Ollama model: {self.config.model_path}")
                self.client.pull(self.config.model_path)

        except ImportError:
            self.logger.error("Ollama not installed. Install with: pip install ollama")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama client: {e}")
            raise

    async def generate_async(self, prompt: str, params: InferenceParams) -> ModelResponse:
        """Ollama를 통한 비동기 텍스트 생성"""
        start_time = time.time()

        def generate_sync():
            generation_start = time.time()

            response = self.client.generate(
                model=self.config.model_path,
                prompt=prompt,
                options={
                    'temperature': params.temperature,
                    'top_p': params.top_p,
                    'top_k': params.top_k,
                    'repeat_penalty': params.repetition_penalty,
                    'num_predict': params.max_new_tokens,
                }
            )

            generation_time = time.time() - generation_start
            generated_text = response['response']

            # 통계 계산 (Ollama 응답에서 제공되는 정보 활용)
            eval_count = response.get('eval_count', 0)
            eval_duration = response.get('eval_duration', 0) / 1e9  # 나노초를 초로 변환
            tokens_per_second = eval_count / eval_duration if eval_duration > 0 else 0

            return generated_text, {
                'input_tokens': response.get('prompt_eval_count', 0),
                'output_tokens': eval_count,
                'total_tokens': response.get('prompt_eval_count', 0) + eval_count,
                'tokens_per_second': tokens_per_second,
                'generation_time': generation_time,
                'memory_usage_mb': self._get_memory_usage(),
                'load_duration': response.get('load_duration', 0) / 1e9,
                'eval_duration': eval_duration
            }

        # 별도 스레드에서 실행
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            generated_text, stats = await loop.run_in_executor(executor, generate_sync)

        latency = time.time() - start_time

        return ModelResponse(
            content=generated_text,
            model=self.config.name,
            generation_stats=stats,
            latency=latency,
            timestamp=datetime.now(),
            params=params,
            metadata={'backend': 'ollama'}
        )

class TGIInterface(BaseModelInterface):
    """Text Generation Inference (TGI) 인터페이스"""

    def _initialize_model(self):
        """TGI 클라이언트 초기화"""
        try:
            from text_generation import AsyncClient

            self.client = AsyncClient(
                base_url=self.config.base_url,
                timeout=60
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.logger.info(f"TGI client initialized for: {self.config.base_url}")

        except ImportError:
            self.logger.error("text-generation not installed. Install with: pip install text-generation")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize TGI client: {e}")
            raise

    async def generate_async(self, prompt: str, params: InferenceParams) -> ModelResponse:
        """TGI를 통한 비동기 텍스트 생성"""
        start_time = time.time()

        try:
            generation_start = time.time()

            response = await self.client.generate(
                prompt=prompt,
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                repetition_penalty=params.repetition_penalty,
                do_sample=params.do_sample,
                return_full_text=False,
                details=True
            )

            generation_time = time.time() - generation_start
            generated_text = response.generated_text

            # TGI 상세 정보에서 통계 추출
            details = response.details
            tokens_per_second = details.generated_tokens / generation_time if generation_time > 0 else 0

            stats = {
                'input_tokens': details.prefill[0].id if details.prefill else 0,
                'output_tokens': details.generated_tokens,
                'total_tokens': (details.prefill[0].id if details.prefill else 0) + details.generated_tokens,
                'tokens_per_second': tokens_per_second,
                'generation_time': generation_time,
                'memory_usage_mb': self._get_memory_usage(),
                'finish_reason': details.finish_reason.name if details.finish_reason else 'unknown'
            }

        except Exception as e:
            self.logger.error(f"TGI generation failed: {e}")
            raise

        latency = time.time() - start_time

        return ModelResponse(
            content=generated_text,
            model=self.config.name,
            generation_stats=stats,
            latency=latency,
            timestamp=datetime.now(),
            params=params,
            metadata={'backend': 'tgi'}
        )

class PerformanceMonitor:
    """성능 모니터링 클래스"""

    def __init__(self):
        self.is_monitoring = False
        self.metrics = []
        self.monitor_thread = None

    def start_monitoring(self):
        """모니터링 시작"""
        self.is_monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def stop_monitoring(self) -> PerformanceMetrics:
        """모니터링 종료 및 결과 반환"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        if not self.metrics:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        # 통계 계산
        latencies = [m['latency'] for m in self.metrics]
        memory_usages = [m['memory'] for m in self.metrics]
        cpu_usages = [m['cpu'] for m in self.metrics]
        gpu_usages = [m['gpu'] for m in self.metrics]

        return PerformanceMetrics(
            tokens_per_second=sum(m.get('tokens_per_second', 0) for m in self.metrics) / len(self.metrics),
            memory_usage_mb=max(memory_usages),
            gpu_utilization=sum(gpu_usages) / len(gpu_usages),
            cpu_utilization=sum(cpu_usages) / len(cpu_usages),
            latency_p50=sorted(latencies)[len(latencies)//2],
            latency_p95=sorted(latencies)[int(len(latencies)*0.95)],
            latency_p99=sorted(latencies)[int(len(latencies)*0.99)],
            throughput=len(self.metrics) / (self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp']) if len(self.metrics) > 1 else 0
        )

    def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                # CPU 사용률
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # 메모리 사용량
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / 1024 / 1024

                # GPU 사용률 (가능한 경우)
                gpu_percent = 0
                if torch.cuda.is_available():
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_percent = utilization.gpu
                    except:
                        gpu_percent = 0

                self.metrics.append({
                    'timestamp': time.time(),
                    'cpu': cpu_percent,
                    'memory': memory_mb,
                    'gpu': gpu_percent,
                    'latency': 0  # 실제 요청에서 업데이트됨
                })

                time.sleep(0.5)  # 0.5초마다 모니터링

            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                break

class ModelManager:
    """모델 관리 클래스"""

    def __init__(self, optimization_config: OptimizationConfig):
        self.optimization_config = optimization_config
        self.interfaces = {}
        self.model_configs = {}

    def register_model(self, name: str, config: ModelConfig):
        """모델 등록"""
        self.model_configs[name] = config

        # 모델 타입에 따라 적절한 인터페이스 생성
        if config.model_type == "transformers":
            interface = TransformersInterface(config, self.optimization_config)
        elif config.model_type == "vllm":
            interface = VLLMInterface(config, self.optimization_config)
        elif config.model_type == "ollama":
            interface = OllamaInterface(config, self.optimization_config)
        elif config.model_type == "tgi":
            interface = TGIInterface(config, self.optimization_config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        self.interfaces[name] = interface
        logging.info(f"Model registered: {name} ({config.model_type})")

    def get_interface(self, model_name: str) -> BaseModelInterface:
        """모델 인터페이스 반환"""
        if model_name not in self.interfaces:
            raise ValueError(f"Model {model_name} not registered")
        return self.interfaces[model_name]

    def list_models(self) -> List[str]:
        """등록된 모델 목록 반환"""
        return list(self.interfaces.keys())

    def get_model_info_all(self) -> Dict[str, Dict[str, Any]]:
        """모든 모델 정보 반환"""
        return {
            name: interface.get_model_info()
            for name, interface in self.interfaces.items()
        }

    async def benchmark_models(self, model_names: List[str], test_prompt: str,
                             params: InferenceParams, iterations: int = 5) -> Dict[str, PerformanceMetrics]:
        """모델 벤치마크 실행"""
        results = {}

        for model_name in model_names:
            if model_name not in self.interfaces:
                continue

            interface = self.interfaces[model_name]
            latencies = []
            tokens_per_second_list = []

            logging.info(f"Benchmarking {model_name}...")

            for i in range(iterations):
                try:
                    response = await interface.generate_async(test_prompt, params)
                    latencies.append(response.latency)
                    tokens_per_second_list.append(response.generation_stats.get('tokens_per_second', 0))
                except Exception as e:
                    logging.error(f"Benchmark failed for {model_name} iteration {i}: {e}")

            if latencies:
                # 성능 메트릭 계산
                avg_latency = sum(latencies) / len(latencies)
                avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list)

                results[model_name] = PerformanceMetrics(
                    tokens_per_second=avg_tokens_per_second,
                    memory_usage_mb=interface._get_memory_usage(),
                    gpu_utilization=0,  # 실제 구현에서는 GPU 모니터링 추가
                    cpu_utilization=0,
                    latency_p50=sorted(latencies)[len(latencies)//2],
                    latency_p95=sorted(latencies)[int(len(latencies)*0.95)],
                    latency_p99=sorted(latencies)[int(len(latencies)*0.99)],
                    throughput=1.0 / avg_latency if avg_latency > 0 else 0
                )

        return results


# model_interface.py 추가 수정사항

import signal
import os
import sys
from contextlib import contextmanager


@contextmanager
def signal_safe_context():
    """신호 처리 안전 컨텍스트"""
    # 메인 스레드에서만 신호 처리 비활성화
    if threading.current_thread() is threading.main_thread():
        old_handler = None
        try:
            # SIGALRM 핸들러 임시 비활성화
            old_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
            yield
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
    else:
        yield


class SignalSafeTransformersInterface(ThreadSafeTransformersInterface):
    """신호 안전 Transformers 인터페이스"""

    def _initialize_model(self):
        """신호 안전 모델 초기화"""
        try:
            with signal_safe_context():
                super()._initialize_model()
        except Exception as e:
            if "signal only works in main thread" in str(e):
                self.logger.warning("Signal handling disabled for thread safety")
                # trust_remote_code 검증 비활성화
                os.environ['HF_TRUST_REMOTE_CODE'] = 'false'
                super()._initialize_model()
            else:
                raise

    def _load_model_with_fallback(self):
        """대체 방법으로 모델 로딩"""
        try:
            # 1차 시도: 일반 로딩
            return self._load_model_primary()
        except Exception as e:
            self.logger.warning(f"Primary loading failed: {e}")

            try:
                # 2차 시도: 신뢰 코드 비활성화
                return self._load_model_no_trust()
            except Exception as e2:
                self.logger.warning(f"No-trust loading failed: {e2}")

                # 3차 시도: CPU 전용
                return self._load_model_cpu_only()

    def _load_model_primary(self):
        """기본 모델 로딩"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs = {
            'trust_remote_code': self.config.trust_remote_code,
            'torch_dtype': getattr(torch, self.config.dtype),
            'device_map': self.config.device if self.config.device != "auto" else "auto",
        }

        # 양자화 설정
        if self.config.load_in_4bit or self.config.load_in_8bit:
            model_kwargs['quantization_config'] = self._get_quantization_config()

        return AutoModelForCausalLM.from_pretrained(
            self.config.model_path, **model_kwargs
        )

    def _load_model_no_trust(self):
        """신뢰 코드 없이 모델 로딩"""
        from transformers import AutoModelForCausalLM

        model_kwargs = {
            'trust_remote_code': False,  # 강제 비활성화
            'torch_dtype': getattr(torch, self.config.dtype),
            'device_map': self.config.device if self.config.device != "auto" else "auto",
        }

        return AutoModelForCausalLM.from_pretrained(
            self.config.model_path, **model_kwargs
        )

    def _load_model_cpu_only(self):
        """CPU 전용 모델 로딩"""
        from transformers import AutoModelForCausalLM

        model_kwargs = {
            'trust_remote_code': False,
            'torch_dtype': torch.float32,  # CPU는 float32
            'device_map': 'cpu',
        }

        self.logger.info("Falling back to CPU-only mode")
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_path, **model_kwargs
        )


class RobustModelManager(ModelManager):
    """강화된 모델 매니저"""

    def __init__(self, optimization_config: OptimizationConfig):
        super().__init__(optimization_config)
        self.error_handler = ErrorHandler(logging.getLogger(__name__))
        self.model_locks = {}  # 모델별 락

    def register_model(self, name: str, config: ModelConfig):
        """강화된 모델 등록"""
        if name in self.model_locks:
            return  # 이미 등록됨

        self.model_locks[name] = threading.Lock()

        try:
            # 모델 타입에 따른 안전한 인터페이스 선택
            if config.model_type == "transformers":
                if threading.current_thread() is threading.main_thread():
                    interface = SignalSafeTransformersInterface(config, self.optimization_config)
                else:
                    interface = ThreadSafeTransformersInterface(config, self.optimization_config)
            elif config.model_type == "vllm":
                interface = VLLMInterface(config, self.optimization_config)
            elif config.model_type == "ollama":
                interface = OllamaInterface(config, self.optimization_config)
            elif config.model_type == "tgi":
                interface = TGIInterface(config, self.optimization_config)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")

            self.interfaces[name] = interface
            logging.info(f"Model registered safely: {name} ({config.model_type})")

        except Exception as e:
            self.error_handler.handle_model_loading_error(name, e)
            raise

    def get_interface(self, model_name: str) -> BaseModelInterface:
        """스레드 안전 인터페이스 반환"""
        if model_name not in self.interfaces:
            raise ValueError(f"Model {model_name} not registered")

        # 모델별 락 사용
        with self.model_locks.get(model_name, threading.Lock()):
            return self.interfaces[model_name]


# 환경 변수 설정으로 일반적인 문제 해결
def setup_safe_environment():
    """안전한 환경 설정"""
    # HuggingFace 관련
    os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    os.environ.setdefault('TRANSFORMERS_CACHE', os.path.expanduser('~/.cache/huggingface/transformers'))

    # PyTorch 관련
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

    # 신호 처리 관련
    if 'TOKENIZERS_PARALLELISM' not in os.environ:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 토크나이저 병렬 처리 비활성화

    # 멀티프로세싱 관련
    if sys.platform.startswith('linux'):
        os.environ.setdefault('OMP_NUM_THREADS', '1')


# 모듈 초기화 시 안전한 환경 설정
setup_safe_environment()


# 사용 예시
if __name__ == "__main__":
    from config import ConfigManager, OptimizationConfig

    # 설정 로드
    config_manager = ConfigManager()
    optimization_config = config_manager.optimization_config

    # 모델 매니저 생성
    model_manager = ModelManager(optimization_config)

    # 테스트용 로컬 모델 등록
    test_config = ModelConfig(
        name="test-model",
        model_path="microsoft/DialoGPT-medium",
        model_type="transformers",
        device="auto",
        dtype="float16"
    )

    async def test_model_interface():
        try:
            model_manager.register_model("test", test_config)
            interface = model_manager.get_interface("test")

            # 모델 정보 출력
            model_info = interface.get_model_info()
            print("모델 정보:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")

            # 테스트 생성
            params = InferenceParams(
                temperature=0.7,
                max_new_tokens=50,
                top_p=0.9
            )

            response = await interface.generate_async("안녕하세요!", params)
            print(f"\n생성 결과:")
            print(f"응답: {response.content}")
            print(f"지연시간: {response.latency:.3f}초")
            print(f"토큰/초: {response.generation_stats.get('tokens_per_second', 0):.1f}")

        except Exception as e:
            print(f"테스트 실패: {e}")

    # 테스트 실행
    print("=== 오픈소스 LLM 인터페이스 테스트 ===")
    asyncio.run(test_model_interface())