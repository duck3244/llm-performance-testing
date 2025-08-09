"""
안전성 강화된 오픈소스 LLM 추론 성능 최적화 테스트 실행기
모든 주요 문제점이 해결된 버전
"""
import asyncio
import json
import time
import torch
import psutil
import gc
import threading
import uuid
import queue
import signal
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging
import warnings
import weakref

# 안전한 환경 설정
def setup_safe_environment():
    """안전한 환경 설정"""
    os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    os.environ.setdefault('TRANSFORMERS_CACHE', os.path.expanduser('~/.cache/huggingface/transformers'))
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if sys.platform.startswith('linux'):
        os.environ.setdefault('OMP_NUM_THREADS', '1')

setup_safe_environment()

from config import ConfigManager, InferenceParams, OptimizationConfig, ModelConfig
from dataset_loader import DatasetManager, TestSample
from evaluator import get_cached_evaluator, evaluate_with_memory_management
from model_interface import ModelManager, PerformanceMetrics

@dataclass
class OptimizationResult:
    """최적화 결과 데이터 클래스"""
    test_id: str
    model_name: str
    dataset_name: str
    best_params: InferenceParams
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_time: float
    hardware_usage: Dict[str, float]
    recommendations: List[str]

@dataclass
class BenchmarkResult:
    """벤치마크 결과 데이터 클래스"""
    test_id: str
    model_name: str
    dataset_name: str
    params: InferenceParams
    performance_metrics: PerformanceMetrics
    evaluation_results: List[Dict[str, Any]]  # 직렬화 가능하도록 수정
    hardware_efficiency: Dict[str, float]
    cost_analysis: Dict[str, float]
    timestamp: datetime

@contextmanager
def signal_safe_context():
    """신호 처리 안전 컨텍스트"""
    if threading.current_thread() is threading.main_thread():
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
            yield
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
    else:
        yield

class ResourceManager:
    """리소스 관리자 - 메모리 누수 방지"""

    def __init__(self):
        self._active_models: Dict[str, weakref.ref] = {}
        self._memory_threshold = 0.9
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def managed_model(self, model_name: str, model_interface):
        """모델 리소스 관리 컨텍스트"""
        self._active_models[model_name] = weakref.ref(model_interface)
        try:
            yield model_interface
        finally:
            self._cleanup_model(model_name)

    def _cleanup_model(self, model_name: str):
        """모델 정리"""
        if model_name in self._active_models:
            model_ref = self._active_models[model_name]
            model = model_ref()
            if model is not None:
                if hasattr(model, 'model') and hasattr(model.model, 'cpu'):
                    try:
                        model.model.cpu()
                        del model.model
                    except:
                        pass
            del self._active_models[model_name]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def check_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 확인"""
        usage = {}

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3

                    usage[f'gpu_{i}'] = {
                        'allocated_gb': allocated,
                        'reserved_gb': reserved,
                        'total_gb': total,
                        'utilization': allocated / total if total > 0 else 0
                    }

                    if allocated / total > self._memory_threshold:
                        self.logger.warning(f"GPU {i} memory usage high: {allocated/total:.1%}")
                except Exception as e:
                    self.logger.warning(f"Failed to get GPU {i} memory info: {e}")

        return usage

    def emergency_cleanup(self):
        """긴급 메모리 정리"""
        self.logger.info("Emergency memory cleanup initiated")

        for model_name in list(self._active_models.keys()):
            self._cleanup_model(model_name)

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass

class ErrorHandler:
    """중앙화된 오류 처리"""

    def __init__(self, logger):
        self.logger = logger
        self.error_counts = {}

    def handle_model_loading_error(self, model_name: str, error: Exception) -> bool:
        """모델 로딩 오류 처리"""
        error_key = f"model_loading_{model_name}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        if self.error_counts[error_key] > 3:
            self.logger.error(f"Model {model_name} failed to load {self.error_counts[error_key]} times. Giving up.")
            return False

        self.logger.warning(f"Model {model_name} loading failed (attempt {self.error_counts[error_key]}): {error}")

        if "CUDA out of memory" in str(error):
            self.logger.info("Suggestion: Try enabling quantization or reducing batch size")
        elif "No module named" in str(error):
            self.logger.info("Suggestion: Install missing dependencies")
        elif "trust_remote_code" in str(error):
            self.logger.info("Suggestion: Set trust_remote_code=True for this model")

        return True

    def handle_inference_error(self, error: Exception, context: Dict[str, Any]) -> Optional[str]:
        """추론 오류 처리"""
        error_str = str(error)

        if "CUDA out of memory" in error_str:
            return "GPU memory insufficient. Try reducing batch size or enable quantization."
        elif "Already borrowed" in error_str:
            return "Thread safety issue detected. Switching to sequential processing."
        elif "signal only works in main thread" in error_str:
            return "Threading issue with transformers. Using fallback mode."
        else:
            self.logger.error(f"Unhandled inference error: {error}")
            return None

class ThreadSafeModelInterface:
    """스레드 안전 모델 인터페이스"""

    def __init__(self, config: ModelConfig, optimization_config: OptimizationConfig):
        self.config = config
        self.optimization_config = optimization_config
        self.logger = logging.getLogger(__name__)

        self._model_lock = threading.Lock()
        self._request_queue = queue.Queue()
        self._response_queues = {}
        self._worker_thread = None
        self._shutdown = False

        self.model = None
        self.tokenizer = None
        self.device = None

        self._initialize_model()
        self._start_worker()

    def _initialize_model(self):
        """모델 초기화"""
        try:
            with signal_safe_context():
                self._load_model()
        except Exception as e:
            if "signal only works in main thread" in str(e):
                self.logger.warning("Signal handling disabled for thread safety")
                os.environ['HF_TRUST_REMOTE_CODE'] = 'false'
                self._load_model()
            else:
                raise

    def _load_model(self):
        """실제 모델 로딩"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            self.logger.info(f"Loading model: {self.config.model_path}")

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="left"
            )

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

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path, **model_kwargs
            )

            self.device = next(self.model.parameters()).device
            self.logger.info(f"Model loaded on device: {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise

    def _start_worker(self):
        """워커 스레드 시작"""
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        """단일 스레드에서 모든 추론 처리"""
        while not self._shutdown:
            try:
                request_id, prompt, params = self._request_queue.get(timeout=1)

                with self._model_lock:
                    result = self._generate_sync(prompt, params)

                if request_id in self._response_queues:
                    self._response_queues[request_id].put(result)

                self._request_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")

    def _generate_sync(self, prompt: str, params: InferenceParams):
        """동기 생성"""
        start_time = time.time()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        input_length = inputs['input_ids'].shape[1]

        generation_kwargs = {
            'max_new_tokens': min(params.max_new_tokens, 512),  # 안전 제한
            'temperature': params.temperature,
            'top_p': params.top_p,
            'top_k': params.top_k,
            'do_sample': params.do_sample,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            generation_start = time.time()
            outputs = self.model.generate(**inputs, **generation_kwargs)
            generation_time = time.time() - generation_start

        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        total_tokens = len(generated_tokens)
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

        return {
            'content': generated_text,
            'generation_stats': {
                'input_tokens': input_length,
                'output_tokens': total_tokens,
                'total_tokens': input_length + total_tokens,
                'tokens_per_second': tokens_per_second,
                'generation_time': generation_time,
                'memory_usage_mb': self._get_memory_usage()
            },
            'latency': time.time() - start_time
        }

    def _get_memory_usage(self) -> float:
        """메모리 사용량 반환"""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024

    async def generate_async(self, prompt: str, params: InferenceParams):
        """비동기 생성"""
        request_id = str(uuid.uuid4())
        response_queue = queue.Queue()
        self._response_queues[request_id] = response_queue

        self._request_queue.put((request_id, prompt, params))

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, response_queue.get)

        del self._response_queues[request_id]
        return result

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        try:
            if self.model is None:
                return {'status': 'not_loaded'}

            model_size = sum(p.numel() for p in self.model.parameters())
            return {
                'model_name': self.config.name,
                'model_path': self.config.model_path,
                'device': str(self.device),
                'parameter_count': model_size,
                'memory_footprint_mb': self._get_memory_usage(),
                'status': 'loaded'
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

class SafeModelManager:
    """안전한 모델 매니저"""

    def __init__(self, optimization_config: OptimizationConfig):
        self.optimization_config = optimization_config
        self.interfaces = {}
        self.model_configs = {}
        self.error_handler = ErrorHandler(logging.getLogger(__name__))
        self.model_locks = {}

    def register_model(self, name: str, config: ModelConfig, max_retries: int = 3):
        """안전한 모델 등록"""
        if name in self.interfaces:
            return

        self.model_locks[name] = threading.Lock()

        for attempt in range(max_retries):
            try:
                if config.model_type == "transformers":
                    interface = ThreadSafeModelInterface(config, self.optimization_config)
                else:
                    # 다른 타입들은 기본 구현 사용
                    interface = ThreadSafeModelInterface(config, self.optimization_config)

                self.interfaces[name] = interface
                self.model_configs[name] = config
                logging.info(f"Model registered successfully: {name}")
                return

            except Exception as e:
                if not self.error_handler.handle_model_loading_error(name, e):
                    break

                if attempt < max_retries - 1:
                    config = self._adjust_config_for_retry(config, e)

        logging.error(f"Failed to register model {name} after {max_retries} attempts")
        raise RuntimeError(f"Cannot register model {name}")

    def _adjust_config_for_retry(self, config: ModelConfig, error: Exception) -> ModelConfig:
        """재시도를 위한 설정 조정"""
        error_str = str(error)

        if "CUDA out of memory" in error_str:
            config.load_in_4bit = True
            config.dtype = "int8"
        elif "trust_remote_code" in error_str:
            config.trust_remote_code = True

        return config

    def get_interface(self, model_name: str):
        """스레드 안전 인터페이스 반환"""
        if model_name not in self.interfaces:
            raise ValueError(f"Model {model_name} not registered")

        with self.model_locks.get(model_name, threading.Lock()):
            return self.interfaces[model_name]

    def list_models(self) -> List[str]:
        """등록된 모델 목록"""
        return list(self.interfaces.keys())

class HardwareMonitor:
    """하드웨어 모니터링 클래스"""

    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.start_time = None

    def start_monitoring(self):
        """모니터링 시작"""
        self.monitoring = True
        self.metrics = []
        self.start_time = time.time()

        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """모니터링 종료"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()

        if not self.metrics:
            return {}

        return self._calculate_summary()

    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()

                metric = {
                    'timestamp': time.time() - self.start_time,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / 1024**3
                }

                if torch.cuda.is_available():
                    try:
                        metric['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
                        metric['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
                    except:
                        pass

                self.metrics.append(metric)
                time.sleep(1)

            except Exception as e:
                logging.error(f"Hardware monitoring error: {e}")
                break

    def _calculate_summary(self) -> Dict[str, float]:
        """모니터링 결과 요약"""
        if not self.metrics:
            return {}

        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]

        summary = {
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'cpu_peak': max(cpu_values),
            'memory_avg': sum(memory_values) / len(memory_values),
            'memory_peak': max(memory_values),
            'duration': self.metrics[-1]['timestamp']
        }

        gpu_memory_values = [m.get('gpu_memory_allocated', 0) for m in self.metrics]
        if any(gpu_memory_values):
            summary.update({
                'gpu_memory_avg': sum(gpu_memory_values) / len(gpu_memory_values),
                'gpu_memory_peak': max(gpu_memory_values)
            })

        estimated_power = 50 + (summary['cpu_avg'] / 100) * 150
        summary['power_consumption'] = estimated_power

        return summary

    def get_usage_summary(self) -> Dict[str, float]:
        """현재 사용량 요약"""
        return self._calculate_summary()

class SafePerformanceOptimizer:
    """안전한 성능 최적화기"""

    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.dataset_manager = DatasetManager()
        self.model_manager = SafeModelManager(self.config_manager.optimization_config)
        self.resource_manager = ResourceManager()
        self.hardware_monitor = HardwareMonitor()

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self.results_dir = Path("optimization_results")
        self.results_dir.mkdir(exist_ok=True)

    def setup_models(self):
        """모델 설정 및 등록"""
        for name, config in self.config_manager.model_configs.items():
            try:
                self.model_manager.register_model(name, config)
                self.logger.info(f"모델 등록됨: {name}")
            except Exception as e:
                self.logger.error(f"모델 {name} 등록 실패: {e}")

    async def optimize_inference_params(self, model_name: str, dataset_name: str,
                                      evaluator_type: str = 'korean_math',
                                      optimization_strategy: str = 'grid_search',
                                      max_trials: int = 20,
                                      num_samples: int = 50) -> OptimizationResult:
        """안전한 파라미터 최적화"""
        test_id = f"opt_{model_name}_{dataset_name}_{int(time.time())}"
        self.logger.info(f"파라미터 최적화 시작: {test_id}")

        start_time = time.time()

        # 메모리 체크
        memory_usage = self.resource_manager.check_memory_usage()
        self.logger.info(f"Initial memory usage: {memory_usage}")

        try:
            # 데이터셋 로드
            samples = await self._load_dataset_for_model(model_name, dataset_name, num_samples)

            # 모델 인터페이스 가져오기
            model_interface = self.model_manager.get_interface(model_name)

            # 리소스 관리 컨텍스트에서 실행
            with self.resource_manager.managed_model(model_name, model_interface):
                if optimization_strategy == 'bayesian':
                    result = await self._safe_bayesian_optimization(
                        model_interface, samples, evaluator_type, max_trials, test_id
                    )
                elif optimization_strategy == 'grid_search':
                    result = await self._grid_search_optimization(
                        model_interface, samples, evaluator_type, test_id
                    )
                else:  # evolutionary
                    result = await self._grid_search_optimization(  # 폴백
                        model_interface, samples, evaluator_type, test_id
                    )

            total_time = time.time() - start_time
            hardware_usage = self.hardware_monitor.get_usage_summary()
            recommendations = self._generate_recommendations(result, hardware_usage)

            optimization_result = OptimizationResult(
                test_id=test_id,
                model_name=model_name,
                dataset_name=dataset_name,
                best_params=result['best_params'],
                best_score=result['best_score'],
                optimization_history=result['history'],
                total_time=total_time,
                hardware_usage=hardware_usage,
                recommendations=recommendations
            )

            self._save_optimization_result(optimization_result)
            self.logger.info(f"최적화 완료: 최고 점수 {result['best_score']:.3f}")

            return optimization_result

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            self.resource_manager.emergency_cleanup()
            raise

    async def _safe_bayesian_optimization(self, model_interface, samples: List[TestSample],
                                        evaluator_type: str, max_trials: int, test_id: str) -> Dict[str, Any]:
        """안전한 베이지안 최적화"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            from skopt.utils import use_named_args
        except ImportError:
            self.logger.warning("scikit-optimize not available, falling back to grid search")
            return await self._grid_search_optimization(model_interface, samples, evaluator_type, test_id)

        dimensions = [
            Real(0.0, 1.0, name='temperature'),
            Real(0.1, 1.0, name='top_p'),
            Integer(1, 100, name='top_k'),
            Integer(50, 512, name='max_new_tokens'),  # 제한된 범위
            Real(1.0, 1.5, name='repetition_penalty')
        ]

        history = []
        best_score = 0
        best_params = None

        @use_named_args(dimensions)
        def objective(**params):
            nonlocal best_score, best_params

            inference_params = InferenceParams(**params)

            # 새 이벤트 루프에서 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                score = loop.run_until_complete(
                    self._evaluate_params_safe(model_interface, samples, evaluator_type, inference_params)
                )
            finally:
                loop.close()

            history.append({
                'params': params,
                'score': score,
                'timestamp': datetime.now().isoformat()
            })

            if score > best_score:
                best_score = score
                best_params = inference_params

            self.logger.info(f"Trial {len(history)}: Score {score:.3f}")
            return -score

        # 스레드풀에서 실행
        def run_optimization():
            return gp_minimize(objective, dimensions, n_calls=max_trials, random_state=42)

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(executor, run_optimization)

        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': history
        }

    async def _grid_search_optimization(self, model_interface, samples: List[TestSample],
                                      evaluator_type: str, test_id: str) -> Dict[str, Any]:
        """그리드 서치 최적화"""
        param_grid = {
            'temperature': [0.0, 0.1, 0.3, 0.7],
            'top_p': [0.1, 0.5, 0.9],
            'top_k': [1, 10, 50],
            'max_new_tokens': [100, 200, 300],
            'repetition_penalty': [1.0, 1.1]
        }

        # 조합 생성 (제한적)
        combinations = []
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        # 최대 20개로 제한
        if len(combinations) > 20:
            import random
            combinations = random.sample(combinations, 20)

        history = []
        best_score = 0
        best_params = None

        for i, param_dict in enumerate(combinations):
            inference_params = InferenceParams(**param_dict)

            self.logger.info(f"그리드 서치 {i+1}/{len(combinations)}: {param_dict}")

            score = await self._evaluate_params_safe(
                model_interface, samples, evaluator_type, inference_params
            )

            history.append({
                'params': param_dict,
                'score': score,
                'timestamp': datetime.now().isoformat()
            })

            if score > best_score:
                best_score = score
                best_params = inference_params

        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': history
        }

    async def _evaluate_params_safe(self, model_interface, samples: List[TestSample],
                                  evaluator_type: str, params: InferenceParams) -> float:
        """안전한 파라미터 평가"""
        try:
            # 작은 배치로 분할 처리
            batch_size = min(5, len(samples))
            total_score = 0
            total_samples = 0

            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i + batch_size]

                # 순차 처리 (메모리 안정성)
                predictions = []
                for sample in batch_samples:
                    try:
                        prompt = self._format_prompt(sample)
                        response = await model_interface.generate_async(prompt, params)
                        predictions.append(response['content'])
                    except Exception as e:
                        self.logger.warning(f"Generation failed: {e}")
                        predictions.append("")  # 빈 예측

                # 평가
                ground_truths = [s.answer for s in batch_samples[:len(predictions)]]

                # 안전한 평가 실행
                metrics = evaluate_with_memory_management(
                    evaluator_type, predictions, ground_truths
                )

                batch_score = metrics.get('accuracy', 0)
                total_score += batch_score * len(predictions)
                total_samples += len(predictions)

                # 주기적 메모리 정리
                if i % (batch_size * 3) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            return total_score / total_samples if total_samples > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Safe parameter evaluation failed: {e}")
            return 0.0

    async def _load_dataset_for_model(self, model_name: str, dataset_name: str, num_samples: int) -> List[TestSample]:
        """모델에 특화된 데이터셋 로드"""
        model_family = self._detect_model_family(model_name)

        test_config = self.config_manager.test_configs.get(dataset_name)
        if not test_config:
            raise ValueError(f"Test config for {dataset_name} not found")

        if model_family:
            loader = self.dataset_manager.get_model_specific_loader(
                model_family, test_config.dataset_path, num_samples=num_samples
            )
        else:
            loader = self.dataset_manager.create_loader(
                'json', dataset_path=test_config.dataset_path, num_samples=num_samples
            )

        return loader.load_samples()

    def _detect_model_family(self, model_name: str) -> Optional[str]:
        """모델 패밀리 감지"""
        model_name_lower = model_name.lower()

        if 'llama' in model_name_lower:
            return 'llama'
        elif 'mistral' in model_name_lower:
            return 'mistral'
        elif 'gemma' in model_name_lower:
            return 'gemma'
        elif 'qwen' in model_name_lower:
            return 'qwen'
        else:
            return None

    def _format_prompt(self, sample: TestSample) -> str:
        """프롬프트 포매팅"""
        return sample.question

    def _generate_recommendations(self, optimization_result: Dict[str, Any],
                                hardware_usage: Dict[str, float]) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        best_params = optimization_result['best_params']

        if best_params.temperature < 0.1:
            recommendations.append("매우 낮은 temperature로 일관성이 높지만 창의성이 제한될 수 있습니다.")
        elif best_params.temperature > 0.8:
            recommendations.append("높은 temperature로 창의적이지만 일관성이 떨어질 수 있습니다.")

        if hardware_usage.get('memory_peak', 0) > 0.9:
            recommendations.append("메모리 사용량이 높습니다. 배치 크기를 줄이거나 양자화를 고려하세요.")

        if optimization_result['best_score'] < 0.7:
            recommendations.append("전체 성능이 낮습니다. 모델 크기를 늘리거나 추가 파인튜닝을 고려하세요.")

        return recommendations

    async def benchmark_model(self, model_name: str, dataset_name: str,
                            params: InferenceParams, num_samples: int = 100,
                            iterations: int = 2) -> BenchmarkResult:
        """모델 벤치마크 실행"""
        test_id = f"bench_{model_name}_{dataset_name}_{int(time.time())}"
        self.logger.info(f"벤치마크 시작: {test_id}")

        self.hardware_monitor.start_monitoring()

        try:
            samples = await self._load_dataset_for_model(model_name, dataset_name, num_samples)
            model_interface = self.model_manager.get_interface(model_name)

            all_results = []
            performance_metrics_list = []

            for i in range(iterations):
                self.logger.info(f"벤치마크 반복 {i+1}/{iterations}")

                # 순차 처리
                responses = []
                start_time = time.time()

                for sample in samples[:50]:  # 최대 50개로 제한
                    try:
                        prompt = self._format_prompt(sample)
                        response = await model_interface.generate_async(prompt, params)
                        responses.append(response)
                    except Exception as e:
                        self.logger.warning(f"Generation failed: {e}")
                        continue

                total_time = time.time() - start_time

                # 평가
                if responses:
                    predictions = [r['content'] for r in responses]
                    ground_truths = [s.answer for s in samples[:len(predictions)]]

                    metrics = evaluate_with_memory_management(
                        'performance', predictions, ground_truths
                    )

                    # 성능 메트릭 계산
                    latencies = [r['latency'] for r in responses]
                    tokens_per_second = [r['generation_stats'].get('tokens_per_second', 0) for r in responses]

                    perf_metrics = PerformanceMetrics(
                        tokens_per_second=sum(tokens_per_second) / len(tokens_per_second),
                        memory_usage_mb=max(r['generation_stats'].get('memory_usage_mb', 0) for r in responses),
                        gpu_utilization=0,
                        cpu_utilization=0,
                        latency_p50=sorted(latencies)[len(latencies)//2],
                        latency_p95=sorted(latencies)[int(len(latencies)*0.95)],
                        latency_p99=sorted(latencies)[int(len(latencies)*0.99)],
                        throughput=len(responses) / total_time if total_time > 0 else 0
                    )

                    performance_metrics_list.append(perf_metrics)

                    # 결과를 직렬화 가능한 형태로 변환
                    serializable_results = []
                    for pred, gt in zip(predictions, ground_truths):
                        serializable_results.append({
                            'prediction': pred,
                            'ground_truth': gt,
                            'score': 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0
                        })
                    all_results.extend(serializable_results)

            hardware_usage = self.hardware_monitor.stop_monitoring()

            # 평균 성능 메트릭
            if performance_metrics_list:
                avg_performance = PerformanceMetrics(
                    tokens_per_second=sum(p.tokens_per_second for p in performance_metrics_list) / len(performance_metrics_list),
                    memory_usage_mb=max(p.memory_usage_mb for p in performance_metrics_list),
                    gpu_utilization=0,
                    cpu_utilization=0,
                    latency_p50=sum(p.latency_p50 for p in performance_metrics_list) / len(performance_metrics_list),
                    latency_p95=sum(p.latency_p95 for p in performance_metrics_list) / len(performance_metrics_list),
                    latency_p99=sum(p.latency_p99 for p in performance_metrics_list) / len(performance_metrics_list),
                    throughput=sum(p.throughput for p in performance_metrics_list) / len(performance_metrics_list)
                )
            else:
                avg_performance = PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0)

            # 효율성 및 비용 분석
            hardware_efficiency = self._calculate_hardware_efficiency(avg_performance, hardware_usage)
            cost_analysis = self._calculate_cost_analysis(avg_performance, hardware_usage)

            benchmark_result = BenchmarkResult(
                test_id=test_id,
                model_name=model_name,
                dataset_name=dataset_name,
                params=params,
                performance_metrics=avg_performance,
                evaluation_results=all_results,
                hardware_efficiency=hardware_efficiency,
                cost_analysis=cost_analysis,
                timestamp=datetime.now()
            )

            self._save_benchmark_result(benchmark_result)
            self.logger.info(f"벤치마크 완료: {avg_performance.tokens_per_second:.1f} tokens/sec")

            return benchmark_result

        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            self.resource_manager.emergency_cleanup()
            raise
        finally:
            self.hardware_monitor.stop_monitoring()

    def _calculate_hardware_efficiency(self, performance: PerformanceMetrics,
                                     hardware_usage: Dict[str, float]) -> Dict[str, float]:
        """하드웨어 효율성 계산"""
        return {
            'tokens_per_watt': performance.tokens_per_second / max(hardware_usage.get('power_consumption', 100), 1),
            'memory_efficiency': performance.tokens_per_second / max(performance.memory_usage_mb, 1),
            'overall_efficiency': performance.tokens_per_second / max(
                hardware_usage.get('power_consumption', 100) * performance.memory_usage_mb / 1000, 1
            )
        }

    def _calculate_cost_analysis(self, performance: PerformanceMetrics,
                               hardware_usage: Dict[str, float]) -> Dict[str, float]:
        """비용 분석"""
        power_cost_per_hour = 0.1
        memory_cost_per_gb_hour = 0.01

        power_consumption = hardware_usage.get('power_consumption', 100)
        memory_usage_gb = performance.memory_usage_mb / 1024

        cost_per_hour = (power_consumption / 1000) * power_cost_per_hour + memory_usage_gb * memory_cost_per_gb_hour
        cost_per_token = cost_per_hour / (performance.tokens_per_second * 3600) if performance.tokens_per_second > 0 else 0

        return {
            'cost_per_hour_usd': cost_per_hour,
            'cost_per_token_usd': cost_per_token,
            'cost_per_1k_tokens_usd': cost_per_token * 1000,
            'estimated_monthly_cost_usd': cost_per_hour * 24 * 30
        }

    def _save_optimization_result(self, result: OptimizationResult):
        """최적화 결과 저장"""
        filename = f"{result.test_id}.json"
        filepath = self.results_dir / filename

        result_dict = {
            'test_id': result.test_id,
            'model_name': result.model_name,
            'dataset_name': result.dataset_name,
            'best_params': asdict(result.best_params),
            'best_score': result.best_score,
            'optimization_history': result.optimization_history,
            'total_time': result.total_time,
            'hardware_usage': result.hardware_usage,
            'recommendations': result.recommendations,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

    def _save_benchmark_result(self, result: BenchmarkResult):
        """벤치마크 결과 저장"""
        filename = f"bench_{result.test_id}.json"
        filepath = self.results_dir / filename

        result_dict = {
            'test_id': result.test_id,
            'model_name': result.model_name,
            'dataset_name': result.dataset_name,
            'params': asdict(result.params),
            'performance_metrics': asdict(result.performance_metrics),
            'hardware_efficiency': result.hardware_efficiency,
            'cost_analysis': result.cost_analysis,
            'timestamp': result.timestamp.isoformat(),
            'evaluation_summary': {
                'total_samples': len(result.evaluation_results),
                'avg_score': sum(r['score'] for r in result.evaluation_results) / len(result.evaluation_results) if result.evaluation_results else 0,
                'accuracy': sum(1 for r in result.evaluation_results if r['score'] > 0.8) / len(result.evaluation_results) if result.evaluation_results else 0
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

# 사용 예시
if __name__ == "__main__":
    async def safe_example():
        print("=== 안전한 오픈소스 LLM 성능 최적화 시스템 ===")

        try:
            # 안전한 최적화기 초기화
            optimizer = SafePerformanceOptimizer()

            # 모델 설정 (안전한 기본값)
            from config import ModelConfig, OptimizationConfig

            safe_config = ModelConfig(
                name="test-model",
                model_path="microsoft/DialoGPT-medium",  # 작은 테스트 모델
                model_type="transformers",
                device="auto",
                dtype="float16",
                load_in_4bit=True,  # 메모리 절약
                trust_remote_code=False  # 보안
            )

            # 모델 등록
            optimizer.model_manager.register_model("test-model", safe_config)

            print("✅ 모델 등록 완료")

            # 안전한 최적화 실행
            result = await optimizer.optimize_inference_params(
                model_name="test-model",
                dataset_name="korean_math",
                optimization_strategy="grid_search",  # 안정적
                max_trials=5,  # 제한적
                num_samples=10  # 작은 시작
            )

            print(f"✅ 최적화 완료!")
            print(f"   최고 점수: {result.best_score:.3f}")
            print(f"   소요 시간: {result.total_time:.1f}초")

            # 메모리 정리
            optimizer.resource_manager.emergency_cleanup()

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    # 안전한 실행
    try:
        asyncio.run(safe_example())
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
    finally:
        # 최종 정리
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass