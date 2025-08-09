"""
안전성 강화된 오픈소스 LLM 최적화 시스템
Optuna 기반으로 재구성, 메모리 안전성 강화
"""
import asyncio
import json
import time
import logging
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Optuna로 대체 (scikit-optimize 문제 해결)
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from config import ConfigManager, InferenceParams, OptimizationConfig, ModelConfig, get_resource_manager
from model_interface import SafeModelManager, ModelResponse, BatchResponse
from dataset_loader import DatasetManager, TestSample


@dataclass
class OptimizationResult:
    """최적화 결과"""
    test_id: str
    model_name: str
    dataset_name: str
    best_params: InferenceParams
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_time: float
    hardware_usage: Dict[str, float]
    recommendations: List[str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    test_id: str
    model_name: str
    dataset_name: str
    params: InferenceParams
    performance_metrics: Dict[str, float]
    evaluation_results: List[Dict[str, Any]]
    hardware_efficiency: Dict[str, float]
    cost_analysis: Dict[str, float]
    timestamp: datetime


class SafeEvaluator:
    """안전한 평가기"""

    def __init__(self, evaluator_type: str = "basic"):
        self.evaluator_type = evaluator_type
        self.logger = logging.getLogger(__name__)

    async def evaluate_responses(self, predictions: List[str],
                                 ground_truths: List[str]) -> Dict[str, float]:
        """응답 평가"""
        if len(predictions) != len(ground_truths):
            self.logger.warning("Predictions and ground truths length mismatch")
            min_len = min(len(predictions), len(ground_truths))
            predictions = predictions[:min_len]
            ground_truths = ground_truths[:min_len]

        if not predictions:
            return {'accuracy': 0.0, 'average_score': 0.0, 'total_samples': 0}

        scores = []
        correct_count = 0

        for pred, gt in zip(predictions, ground_truths):
            try:
                score = self._evaluate_single(pred, gt)
                scores.append(score)
                if score > 0.8:  # 임계값
                    correct_count += 1
            except Exception as e:
                self.logger.warning(f"Evaluation error: {e}")
                scores.append(0.0)

        return {
            'accuracy': correct_count / len(predictions) if predictions else 0.0,
            'average_score': np.mean(scores) if scores else 0.0,
            'total_samples': len(predictions),
            'correct_count': correct_count
        }

    def _evaluate_single(self, prediction: str, ground_truth: str) -> float:
        """단일 평가"""
        if not prediction.strip() or not ground_truth.strip():
            return 0.0

        if self.evaluator_type == "exact_match":
            return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0
        elif self.evaluator_type == "contains":
            return 1.0 if ground_truth.strip().lower() in prediction.strip().lower() else 0.0
        else:  # similarity
            return self._calculate_similarity(prediction, ground_truth)

    def _calculate_similarity(self, pred: str, gt: str) -> float:
        """텍스트 유사도 계산"""
        try:
            import difflib
            return difflib.SequenceMatcher(None, pred.lower(), gt.lower()).ratio()
        except:
            return 0.0


class HardwareMonitor:
    """하드웨어 모니터링"""

    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.lock = threading.Lock()

    def start_monitoring(self):
        """모니터링 시작"""
        with self.lock:
            self.monitoring = True
            self.metrics = []

        def monitor_loop():
            import psutil
            while self.monitoring:
                try:
                    # 시스템 메트릭 수집
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()

                    metric = {
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_used_gb': memory.used / 1024 ** 3
                    }

                    # GPU 메트릭 (가능한 경우)
                    try:
                        import torch
                        if torch.cuda.is_available():
                            metric['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024 ** 3
                            metric['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024 ** 3
                    except:
                        pass

                    with self.lock:
                        self.metrics.append(metric)

                    time.sleep(5)  # 5초마다 수집

                except Exception as e:
                    logging.error(f"Monitoring error: {e}")
                    break

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """모니터링 종료 및 결과 반환"""
        with self.lock:
            self.monitoring = False

            if not self.metrics:
                return {}

            # 통계 계산
            cpu_values = [m['cpu_percent'] for m in self.metrics]
            memory_values = [m['memory_percent'] for m in self.metrics]

            result = {
                'cpu_avg': np.mean(cpu_values),
                'cpu_peak': np.max(cpu_values),
                'memory_avg': np.mean(memory_values),
                'memory_peak': np.max(memory_values),
                'duration': self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp']
            }

            # GPU 메트릭 (있는 경우)
            gpu_values = [m.get('gpu_memory_allocated', 0) for m in self.metrics]
            if any(gpu_values):
                result['gpu_memory_avg'] = np.mean(gpu_values)
                result['gpu_memory_peak'] = np.max(gpu_values)

            return result


class SafeOptimizer:
    """안전한 성능 최적화기"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.dataset_manager = DatasetManager()
        self.model_manager = SafeModelManager(self.config_manager.optimization_config)
        self.resource_manager = get_resource_manager()
        self.hardware_monitor = HardwareMonitor()
        self.evaluator = SafeEvaluator()

        self.logger = logging.getLogger(__name__)
        self.results_dir = Path("optimization_results")
        self.results_dir.mkdir(exist_ok=True)

        # 안전성 설정
        self.max_concurrent_trials = 2  # 동시 시도 수 제한
        self.memory_check_interval = 30  # 30초마다 메모리 체크
        self.max_memory_usage = 0.9  # 90% 메모리 사용률 제한

    async def optimize_parameters(self,
                                  model_name: str,
                                  dataset_name: str,
                                  evaluator_type: str = "similarity",
                                  optimization_strategy: str = "optuna",
                                  max_trials: int = 20,
                                  num_samples: int = 50,
                                  timeout_seconds: int = 3600) -> OptimizationResult:
        """안전한 파라미터 최적화"""

        if not OPTUNA_AVAILABLE and optimization_strategy == "optuna":
            self.logger.warning("Optuna not available, falling back to grid search")
            optimization_strategy = "grid_search"

        test_id = f"opt_{model_name}_{dataset_name}_{int(time.time())}"
        self.logger.info(f"Starting optimization: {test_id}")

        start_time = time.time()
        self.hardware_monitor.start_monitoring()

        try:
            # 데이터셋 로드
            samples = await self._load_dataset(model_name, dataset_name, num_samples)
            if not samples:
                raise ValueError(f"No samples loaded from dataset {dataset_name}")

            # 모델 등록
            model_config = self.config_manager.get_model_config(model_name)
            if not model_config:
                raise ValueError(f"Model configuration not found: {model_name}")

            self.model_manager.register_model(model_name, model_config)

            # 최적화 실행
            if optimization_strategy == "optuna":
                result = await self._optuna_optimization(
                    model_name, samples, evaluator_type, max_trials, timeout_seconds
                )
            else:
                result = await self._grid_search_optimization(
                    model_name, samples, evaluator_type
                )

            # 하드웨어 사용량 및 추천사항
            hardware_usage = self.hardware_monitor.stop_monitoring()
            recommendations = self._generate_recommendations(result, hardware_usage)

            # 결과 생성
            optimization_result = OptimizationResult(
                test_id=test_id,
                model_name=model_name,
                dataset_name=dataset_name,
                best_params=result['best_params'],
                best_score=result['best_score'],
                optimization_history=result['history'],
                total_time=time.time() - start_time,
                hardware_usage=hardware_usage,
                recommendations=recommendations
            )

            # 결과 저장
            self._save_optimization_result(optimization_result)

            self.logger.info(f"Optimization completed: best score {result['best_score']:.3f}")
            return optimization_result

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
        finally:
            # 정리
            self.hardware_monitor.stop_monitoring()
            if model_name in self.model_manager.models:
                self.model_manager.unregister_model(model_name)

            # 메모리 정리
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    async def _optuna_optimization(self,
                                   model_name: str,
                                   samples: List[TestSample],
                                   evaluator_type: str,
                                   max_trials: int,
                                   timeout_seconds: int) -> Dict[str, Any]:
        """Optuna 기반 최적화"""
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not available")

        # 안전한 최적화 범위 설정
        def objective(trial):
            # 파라미터 제안
            temperature = trial.suggest_float('temperature', 0.0, 1.0)
            top_p = trial.suggest_float('top_p', 0.1, 1.0)
            top_k = trial.suggest_int('top_k', 1, 100)
            max_new_tokens = trial.suggest_int('max_new_tokens', 50, 512)
            repetition_penalty = trial.suggest_float('repetition_penalty', 1.0, 1.5)

            params = InferenceParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty
            )

            # 평가 실행
            try:
                score = asyncio.run(self._evaluate_params(model_name, samples, params, evaluator_type))
                return score
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return 0.0

        # Optuna 스터디 생성
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )

        # 최적화 실행
        study.optimize(
            objective,
            n_trials=max_trials,
            timeout=timeout_seconds,
            n_jobs=1  # 메모리 안전성을 위해 단일 작업
        )

        # 결과 정리
        best_trial = study.best_trial
        best_params = InferenceParams(**best_trial.params)

        # 히스토리 생성
        history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial_number': trial.number,
                    'params': trial.params,
                    'score': trial.value,
                    'timestamp': datetime.now().isoformat()
                })

        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'history': history,
            'study_info': {
                'n_trials': len(study.trials),
                'best_trial_number': best_trial.number
            }
        }

    async def _grid_search_optimization(self,
                                        model_name: str,
                                        samples: List[TestSample],
                                        evaluator_type: str) -> Dict[str, Any]:
        """안전한 그리드 서치 최적화"""

        # 제한된 파라미터 그리드
        param_grid = {
            'temperature': [0.0, 0.1, 0.3, 0.7],
            'top_p': [0.3, 0.7, 0.9],
            'top_k': [10, 30, 50],
            'max_new_tokens': [100, 200, 300],
            'repetition_penalty': [1.0, 1.1]
        }

        # 조합 생성 (최대 20개로 제한)
        import itertools
        combinations = []
        for temp in param_grid['temperature']:
            for top_p in param_grid['top_p']:
                for top_k in param_grid['top_k']:
                    for max_tokens in param_grid['max_new_tokens']:
                        for rep_penalty in param_grid['repetition_penalty']:
                            combinations.append({
                                'temperature': temp,
                                'top_p': top_p,
                                'top_k': top_k,
                                'max_new_tokens': max_tokens,
                                'repetition_penalty': rep_penalty
                            })

        # 조합 수 제한
        if len(combinations) > 20:
            import random
            combinations = random.sample(combinations, 20)

        # 각 조합 평가
        results = []
        best_score = 0
        best_params = None

        for i, param_dict in enumerate(combinations):
            self.logger.info(f"Grid search {i + 1}/{len(combinations)}")

            params = InferenceParams(**param_dict)

            try:
                score = await self._evaluate_params(model_name, samples, params, evaluator_type)

                results.append({
                    'trial_number': i,
                    'params': param_dict,
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                })

                if score > best_score:
                    best_score = score
                    best_params = params

                # 메모리 체크
                if i % 5 == 0:
                    self._check_memory_usage()

            except Exception as e:
                self.logger.warning(f"Grid search trial {i} failed: {e}")
                results.append({
                    'trial_number': i,
                    'params': param_dict,
                    'score': 0.0,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

        if best_params is None:
            # 기본값 사용
            best_params = InferenceParams()
            best_score = 0.0

        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': results
        }

    async def _evaluate_params(self,
                               model_name: str,
                               samples: List[TestSample],
                               params: InferenceParams,
                               evaluator_type: str) -> float:
        """파라미터 평가"""

        model = self.model_manager.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not available")

        # 샘플 수 제한 (메모리 안전성)
        max_eval_samples = min(len(samples), 20)
        eval_samples = samples[:max_eval_samples]

        # 평가자 설정
        evaluator = SafeEvaluator(evaluator_type)

        # 생성 및 평가
        try:
            predictions = []
            ground_truths = []

            # 순차 처리 (메모리 안전성)
            for sample in eval_samples:
                try:
                    response = await model.generate_async(sample.question, params)
                    predictions.append(response.content)
                    ground_truths.append(sample.answer)
                except Exception as e:
                    self.logger.warning(f"Generation failed for sample: {e}")
                    predictions.append("")
                    ground_truths.append(sample.answer)

            # 평가
            metrics = await evaluator.evaluate_responses(predictions, ground_truths)
            return metrics.get('average_score', 0.0)

        except Exception as e:
            self.logger.error(f"Parameter evaluation failed: {e}")
            return 0.0

    async def _load_dataset(self, model_name: str, dataset_name: str, num_samples: int) -> List[TestSample]:
        """데이터셋 로드"""
        try:
            # 모델 패밀리 감지
            model_family = self._detect_model_family(model_name)

            # 테스트 설정 가져오기
            test_config = self.config_manager.test_configs.get(dataset_name)
            if not test_config:
                raise ValueError(f"Test config for {dataset_name} not found")

            # 데이터셋 로더 생성
            if model_family:
                loader = self.dataset_manager.get_model_specific_loader(
                    model_family,
                    test_config.dataset_path,
                    num_samples=min(num_samples, 100)  # 안전 제한
                )
            else:
                loader = self.dataset_manager.create_loader(
                    'json',
                    dataset_path=test_config.dataset_path,
                    num_samples=min(num_samples, 100)
                )

            samples = loader.load_samples()
            self.logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
            return samples

        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return []

    def _detect_model_family(self, model_name: str) -> Optional[str]:
        """모델 패밀리 감지"""
        name_lower = model_name.lower()

        if 'llama' in name_lower:
            if '3.2' in name_lower:
                return 'llama3.2'
            elif '3.1' in name_lower:
                return 'llama3.1'
            elif '3' in name_lower:
                return 'llama3'
            else:
                return 'llama2'
        elif 'qwen' in name_lower:
            if '2.5' in name_lower:
                return 'qwen2.5'
            else:
                return 'qwen1.5'
        elif 'mistral' in name_lower:
            return 'mistral'
        elif 'gemma' in name_lower:
            return 'gemma'

        return None

    def _check_memory_usage(self):
        """메모리 사용량 체크"""
        try:
            memory_usage = self.resource_manager.get_memory_usage()
            gpu_utilization = memory_usage.get('gpu_utilization', 0)

            if gpu_utilization > self.max_memory_usage:
                self.logger.warning(f"High memory usage: {gpu_utilization:.1%}")

                # 강제 정리
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 재확인
                memory_usage = self.resource_manager.get_memory_usage()
                gpu_utilization = memory_usage.get('gpu_utilization', 0)

                if gpu_utilization > 0.95:
                    raise RuntimeError("Memory usage too high, stopping optimization")

        except Exception as e:
            self.logger.warning(f"Memory check failed: {e}")

    def _generate_recommendations(self, result: Dict[str, Any], hardware_usage: Dict[str, float]) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        best_params = result['best_params']

        # 파라미터 기반 추천
        if best_params.temperature < 0.1:
            recommendations.append("매우 낮은 temperature로 일관성이 높지만 창의성이 제한될 수 있습니다.")
        elif best_params.temperature > 0.8:
            recommendations.append("높은 temperature로 창의적이지만 일관성이 떨어질 수 있습니다.")

        # 하드웨어 기반 추천
        if hardware_usage.get('memory_peak', 0) > 80:
            recommendations.append("메모리 사용량이 높습니다. 배치 크기를 줄이거나 양자화를 고려하세요.")

        if hardware_usage.get('cpu_peak', 0) > 90:
            recommendations.append("CPU 사용률이 높습니다. 동시 처리 수를 줄여보세요.")

        # 성능 기반 추천
        if result['best_score'] < 0.7:
            recommendations.append("전체 성능이 낮습니다. 모델 크기를 늘리거나 다른 모델을 고려하세요.")
        elif result['best_score'] > 0.9:
            recommendations.append("매우 좋은 성능입니다. 이 설정을 프로덕션에 적용하는 것을 권장합니다.")

        return recommendations

    def _save_optimization_result(self, result: OptimizationResult):
        """최적화 결과 저장"""
        try:
            filename = f"{result.test_id}.json"
            filepath = self.results_dir / filename

            # 직렬화 가능한 형태로 변환
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
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0'
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Results saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    async def benchmark_model(self,
                              model_name: str,
                              dataset_name: str,
                              params: InferenceParams,
                              num_samples: int = 50,
                              iterations: int = 1) -> BenchmarkResult:
        """안전한 모델 벤치마크"""

        test_id = f"bench_{model_name}_{dataset_name}_{int(time.time())}"
        self.logger.info(f"Starting benchmark: {test_id}")

        self.hardware_monitor.start_monitoring()
        start_time = time.time()

        try:
            # 데이터셋 로드
            samples = await self._load_dataset(model_name, dataset_name, num_samples)
            if not samples:
                raise ValueError(f"No samples loaded from dataset {dataset_name}")

            # 모델 등록
            model_config = self.config_manager.get_model_config(model_name)
            if not model_config:
                raise ValueError(f"Model configuration not found: {model_name}")

            self.model_manager.register_model(model_name, model_config)
            model = self.model_manager.get_model(model_name)

            # 벤치마크 실행
            all_responses = []
            all_evaluations = []

            for iteration in range(iterations):
                self.logger.info(f"Benchmark iteration {iteration + 1}/{iterations}")

                # 배치 생성
                prompts = [sample.question for sample in samples]
                batch_response = await model.generate_batch_async(prompts, params)

                # 평가
                predictions = [r.content for r in batch_response.responses]
                ground_truths = [samples[i].answer for i in range(len(predictions))]

                evaluator = SafeEvaluator("similarity")
                eval_metrics = await evaluator.evaluate_responses(predictions, ground_truths)

                all_responses.extend(batch_response.responses)
                all_evaluations.append(eval_metrics)

                # 메모리 체크
                self._check_memory_usage()

            # 성능 메트릭 계산
            performance_metrics = self._calculate_performance_metrics(all_responses, all_evaluations)

            # 하드웨어 사용량
            hardware_usage = self.hardware_monitor.stop_monitoring()

            # 효율성 및 비용 분석
            hardware_efficiency = self._calculate_hardware_efficiency(performance_metrics, hardware_usage)
            cost_analysis = self._calculate_cost_analysis(performance_metrics, hardware_usage)

            # 평가 결과 (직렬화 가능한 형태)
            evaluation_results = []
            for i, response in enumerate(all_responses[:len(samples)]):
                evaluation_results.append({
                    'prediction': response.content,
                    'ground_truth': samples[i].answer,
                    'latency': response.latency,
                    'tokens_per_second': response.generation_stats.get('tokens_per_second', 0)
                })

            benchmark_result = BenchmarkResult(
                test_id=test_id,
                model_name=model_name,
                dataset_name=dataset_name,
                params=params,
                performance_metrics=performance_metrics,
                evaluation_results=evaluation_results,
                hardware_efficiency=hardware_efficiency,
                cost_analysis=cost_analysis,
                timestamp=datetime.now()
            )

            # 결과 저장
            self._save_benchmark_result(benchmark_result)

            self.logger.info(f"Benchmark completed: {performance_metrics.get('tokens_per_second', 0):.1f} tokens/sec")
            return benchmark_result

        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            raise
        finally:
            # 정리
            self.hardware_monitor.stop_monitoring()
            if model_name in self.model_manager.models:
                self.model_manager.unregister_model(model_name)
            gc.collect()

    def _calculate_performance_metrics(self, responses: List[ModelResponse],
                                       evaluations: List[Dict[str, float]]) -> Dict[str, float]:
        """성능 메트릭 계산"""
        if not responses:
            return {}

        # 지연시간 통계
        latencies = [r.latency for r in responses if r.latency > 0]
        tokens_per_second = [r.generation_stats.get('tokens_per_second', 0) for r in responses]
        memory_usage = [r.generation_stats.get('memory_usage_mb', 0) for r in responses]

        metrics = {
            'tokens_per_second': np.mean(tokens_per_second) if tokens_per_second else 0,
            'latency_avg': np.mean(latencies) if latencies else 0,
            'latency_p50': np.percentile(latencies, 50) if latencies else 0,
            'latency_p95': np.percentile(latencies, 95) if latencies else 0,
            'latency_p99': np.percentile(latencies, 99) if latencies else 0,
            'memory_usage_mb': np.max(memory_usage) if memory_usage else 0,
            'throughput': len(responses) / sum(latencies) if latencies else 0
        }

        # 평가 메트릭
        if evaluations:
            accuracy_scores = [e.get('accuracy', 0) for e in evaluations]
            avg_scores = [e.get('average_score', 0) for e in evaluations]

            metrics.update({
                'accuracy': np.mean(accuracy_scores),
                'average_score': np.mean(avg_scores)
            })

        return metrics

    def _calculate_hardware_efficiency(self, performance: Dict[str, float],
                                       hardware: Dict[str, float]) -> Dict[str, float]:
        """하드웨어 효율성 계산"""
        tokens_per_sec = performance.get('tokens_per_second', 0)
        memory_mb = performance.get('memory_usage_mb', 1)

        return {
            'tokens_per_mb': tokens_per_sec / memory_mb if memory_mb > 0 else 0,
            'memory_efficiency': tokens_per_sec / (memory_mb / 1024) if memory_mb > 0 else 0,
            'cpu_efficiency': tokens_per_sec / max(hardware.get('cpu_avg', 1), 1),
            'overall_efficiency': tokens_per_sec / max(memory_mb * hardware.get('cpu_avg', 1) / 1000, 1)
        }

    def _calculate_cost_analysis(self, performance: Dict[str, float],
                                 hardware: Dict[str, float]) -> Dict[str, float]:
        """비용 분석"""
        # 간단한 비용 모델 (실제로는 더 복잡)
        memory_gb = performance.get('memory_usage_mb', 0) / 1024
        duration_hours = hardware.get('duration', 3600) / 3600

        # 가상의 비용 (실제 클라우드 비용으로 대체 가능)
        gpu_cost_per_hour = 0.5  # $0.5/hour for GPU
        memory_cost_per_gb_hour = 0.01  # $0.01/GB/hour

        total_cost = (gpu_cost_per_hour + memory_gb * memory_cost_per_gb_hour) * duration_hours
        tokens_generated = performance.get('tokens_per_second', 0) * duration_hours * 3600

        return {
            'cost_per_hour_usd': gpu_cost_per_hour + memory_gb * memory_cost_per_gb_hour,
            'cost_per_token_usd': total_cost / tokens_generated if tokens_generated > 0 else 0,
            'cost_per_1k_tokens_usd': (total_cost / tokens_generated * 1000) if tokens_generated > 0 else 0,
            'estimated_monthly_cost_usd': (gpu_cost_per_hour + memory_gb * memory_cost_per_gb_hour) * 24 * 30
        }

    def _save_benchmark_result(self, result: BenchmarkResult):
        """벤치마크 결과 저장"""
        try:
            filename = f"bench_{result.test_id}.json"
            filepath = self.results_dir / filename

            result_dict = {
                'test_id': result.test_id,
                'model_name': result.model_name,
                'dataset_name': result.dataset_name,
                'params': asdict(result.params),
                'performance_metrics': result.performance_metrics,
                'hardware_efficiency': result.hardware_efficiency,
                'cost_analysis': result.cost_analysis,
                'timestamp': result.timestamp.isoformat(),
                'evaluation_summary': {
                    'total_samples': len(result.evaluation_results),
                    'avg_latency': result.performance_metrics.get('latency_avg', 0),
                    'tokens_per_second': result.performance_metrics.get('tokens_per_second', 0)
                },
                'version': '2.0.0'
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Benchmark results saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save benchmark results: {e}")


# 사용 예시
if __name__ == "__main__":
    import asyncio


    async def test_safe_optimizer():
        print("=== 안전한 최적화 시스템 테스트 ===")

        try:
            optimizer = SafeOptimizer()

            # 간단한 최적화 테스트
            result = await optimizer.optimize_parameters(
                model_name="qwen2.5-7b",
                dataset_name="korean_math",
                optimization_strategy="grid_search",  # 안전한 방법
                max_trials=5,
                num_samples=10
            )

            print(f"✅ 최적화 완료!")
            print(f"   최고 점수: {result.best_score:.3f}")
            print(f"   소요 시간: {result.total_time:.1f}초")
            print(f"   추천사항: {len(result.recommendations)}개")

            for rec in result.recommendations:
                print(f"   💡 {rec}")

        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 정리
            from safe_config import cleanup_resources
            cleanup_resources()


    # 테스트 실행
    asyncio.run(test_safe_optimizer())