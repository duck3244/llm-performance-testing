"""
오픈소스 LLM 추론 성능 최적화 메인 테스트 실행기
"""
import asyncio
import json
import time
import torch
import psutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import numpy as np

from config import ConfigManager, InferenceParams, OptimizationConfig
from dataset_loader import DatasetManager, TestSample
from evaluator import EvaluatorFactory, EvaluationResult
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
    evaluation_results: List[EvaluationResult]
    hardware_efficiency: Dict[str, float]
    cost_analysis: Dict[str, float]
    timestamp: datetime

class PerformanceOptimizer:
    """성능 최적화 메인 클래스"""

    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.dataset_manager = DatasetManager()
        self.model_manager = ModelManager(self.config_manager.optimization_config)

        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # 결과 저장 디렉토리 생성
        self.results_dir = Path("optimization_results")
        self.results_dir.mkdir(exist_ok=True)

        # 하드웨어 모니터링
        self.hardware_monitor = HardwareMonitor()

    def setup_models(self):
        """모델 설정 및 등록"""
        for name, config in self.config_manager.model_configs.items():
            try:
                self.model_manager.register_model(name, config)
                self.logger.info(f"모델 등록됨: {name} ({config.model_type})")
            except Exception as e:
                self.logger.error(f"모델 {name} 등록 실패: {e}")

    async def optimize_inference_params(self, model_name: str, dataset_name: str,
                                      evaluator_type: str = 'korean_math',
                                      optimization_strategy: str = 'bayesian',
                                      max_trials: int = 50,
                                      num_samples: int = 100) -> OptimizationResult:
        """추론 파라미터 최적화"""
        test_id = f"opt_{model_name}_{dataset_name}_{int(time.time())}"
        self.logger.info(f"파라미터 최적화 시작: {test_id}")

        start_time = time.time()

        # 데이터셋 로드
        samples = await self._load_dataset_for_model(model_name, dataset_name, num_samples)

        # 모델 인터페이스 가져오기
        model_interface = self.model_manager.get_interface(model_name)

        # 평가자 생성
        evaluator = EvaluatorFactory.create_evaluator(evaluator_type)

        # 최적화 전략에 따른 실행
        if optimization_strategy == 'bayesian':
            result = await self._bayesian_optimization(
                model_interface, samples, evaluator, max_trials, test_id
            )
        elif optimization_strategy == 'grid_search':
            result = await self._grid_search_optimization(
                model_interface, samples, evaluator, test_id
            )
        elif optimization_strategy == 'evolutionary':
            result = await self._evolutionary_optimization(
                model_interface, samples, evaluator, max_trials, test_id
            )
        else:
            raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")

        total_time = time.time() - start_time

        # 하드웨어 사용량 분석
        hardware_usage = self.hardware_monitor.get_usage_summary()

        # 추천사항 생성
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

        # 결과 저장
        self._save_optimization_result(optimization_result)

        self.logger.info(f"최적화 완료: 최고 점수 {result['best_score']:.3f}")

        return optimization_result

    async def _bayesian_optimization(self, model_interface, samples: List[TestSample],
                                   evaluator, max_trials: int, test_id: str) -> Dict[str, Any]:
        """베이지안 최적화"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            from skopt.utils import use_named_args
        except ImportError:
            self.logger.error("scikit-optimize not installed. Install with: pip install scikit-optimize")
            return await self._grid_search_optimization(model_interface, samples, evaluator, test_id)

        # 검색 공간 정의
        dimensions = [
            Real(0.0, 1.0, name='temperature'),
            Real(0.1, 1.0, name='top_p'),
            Integer(1, 100, name='top_k'),
            Integer(50, 1024, name='max_new_tokens'),
            Real(1.0, 1.5, name='repetition_penalty')
        ]

        history = []
        best_score = 0
        best_params = None

        @use_named_args(dimensions)
        async def objective(**params):
            nonlocal best_score, best_params

            inference_params = InferenceParams(**params)

            # 평가 실행
            score = await self._evaluate_params(
                model_interface, samples, evaluator, inference_params
            )

            history.append({
                'params': params,
                'score': score,
                'timestamp': datetime.now().isoformat()
            })

            if score > best_score:
                best_score = score
                best_params = inference_params

            self.logger.info(f"점수: {score:.3f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': history
        }

    async def _evolutionary_optimization(self, model_interface, samples: List[TestSample],
                                       evaluator, max_trials: int, test_id: str) -> Dict[str, Any]:
        """진화 알고리즘 최적화"""
        population_size = 10
        generations = max_trials // population_size

        # 초기 개체군 생성
        population = self._generate_initial_population(population_size)
        history = []
        best_score = 0
        best_params = None

        for generation in range(generations):
            self.logger.info(f"진화 세대 {generation + 1}/{generations}")

            # 개체군 평가
            scores = []
            for individual in population:
                inference_params = InferenceParams(**individual)
                score = await self._evaluate_params(
                    model_interface, samples, evaluator, inference_params
                )
                scores.append(score)

                history.append({
                    'params': individual,
                    'score': score,
                    'generation': generation,
                    'timestamp': datetime.now().isoformat()
                })

                if score > best_score:
                    best_score = score
                    best_params = inference_params

            # 선택, 교차, 변이
            population = self._evolve_population(population, scores)

        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': history
        }

    def _generate_initial_population(self, size: int) -> List[Dict[str, Any]]:
        """초기 개체군 생성"""
        population = []
        param_ranges = {
            'temperature': (0.0, 1.0),
            'top_p': (0.1, 1.0),
            'top_k': (1, 100),
            'max_new_tokens': (50, 1024),
            'repetition_penalty': (1.0, 1.5)
        }

        for _ in range(size):
            individual = {}
            for param, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int):
                    individual[param] = np.random.randint(min_val, max_val + 1)
                else:
                    individual[param] = np.random.uniform(min_val, max_val)
            population.append(individual)

        return population

    def _evolve_population(self, population: List[Dict[str, Any]], scores: List[float]) -> List[Dict[str, Any]]:
        """개체군 진화"""
        # 상위 50% 선택
        sorted_indices = np.argsort(scores)[::-1]
        elite_size = len(population) // 2
        elite = [population[i] for i in sorted_indices[:elite_size]]

        # 새로운 개체군 생성
        new_population = elite.copy()

        while len(new_population) < len(population):
            # 부모 선택
            parent1, parent2 = np.random.choice(elite, 2, replace=False)

            # 교차
            child = self._crossover(parent1, parent2)

            # 변이
            child = self._mutate(child)

            new_population.append(child)

        return new_population

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """교차 연산"""
        child = {}
        for key in parent1.keys():
            if np.random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """변이 연산"""
        mutation_rate = 0.1
        param_ranges = {
            'temperature': (0.0, 1.0),
            'top_p': (0.1, 1.0),
            'top_k': (1, 100),
            'max_new_tokens': (50, 1024),
            'repetition_penalty': (1.0, 1.5)
        }

        mutated = individual.copy()
        for param, (min_val, max_val) in param_ranges.items():
            if np.random.random() < mutation_rate:
                if isinstance(min_val, int):
                    mutated[param] = np.random.randint(min_val, max_val + 1)
                else:
                    mutated[param] = np.random.uniform(min_val, max_val)

        return mutated

    def _sample_param_combinations(self, param_grid: Dict[str, List], max_combinations: int) -> List[Dict[str, Any]]:
        """파라미터 조합 샘플링"""
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        all_combinations = list(itertools.product(*values))

        if len(all_combinations) <= max_combinations:
            combinations = all_combinations
        else:
            combinations = np.random.choice(
                len(all_combinations),
                max_combinations,
                replace=False
            )
            combinations = [all_combinations[i] for i in combinations]

        return [dict(zip(keys, combo)) for combo in combinations]

    async def _evaluate_params(self, model_interface, samples: List[TestSample],
                             evaluator, params: InferenceParams) -> float:
        """파라미터 평가"""
        try:
            # 샘플 프롬프트 생성
            prompts = [self._format_prompt(sample) for sample in samples]

            # 배치 생성 실행
            batch_response = await model_interface.generate_batch_async(prompts, params, max_concurrent=2)

            # 평가 실행
            predictions = [r.content for r in batch_response.responses]
            ground_truths = [s.answer for s in samples[:len(predictions)]]
            generation_stats = [r.generation_stats for r in batch_response.responses]

            metrics = evaluator.evaluate_batch(
                predictions, ground_truths, generation_stats
            )

            # 성능과 정확도를 결합한 점수 계산
            accuracy = metrics.get('accuracy', 0)
            avg_latency = metrics.get('avg_latency', 1)
            token_efficiency = metrics.get('avg_token_efficiency', 0.5)

            # 가중 점수 (정확도 70%, 속도 20%, 효율성 10%)
            score = accuracy * 0.7 + (1.0 / avg_latency) * 0.2 + token_efficiency * 0.1

            return score

        except Exception as e:
            self.logger.error(f"파라미터 평가 실패: {e}")
            return 0.0

    async def _load_dataset_for_model(self, model_name: str, dataset_name: str, num_samples: int) -> List[TestSample]:
        """모델에 특화된 데이터셋 로드"""
        # 모델 패밀리 감지
        model_family = self._detect_model_family(model_name)

        # 테스트 설정 가져오기
        test_config = self.config_manager.test_configs.get(dataset_name)
        if not test_config:
            raise ValueError(f"Test config for {dataset_name} not found")

        # 모델별 특화 로더 사용
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
        """프롬프트 포매팅 (이미 모델별로 포맷팅됨)"""
        return sample.question

    def _generate_recommendations(self, optimization_result: Dict[str, Any],
                                hardware_usage: Dict[str, float]) -> List[str]:
        """최적화 결과 기반 추천사항 생성"""
        recommendations = []
        best_params = optimization_result['best_params']

        # 파라미터 기반 추천
        if best_params.temperature < 0.1:
            recommendations.append("매우 낮은 temperature로 일관성이 높지만 창의성이 제한될 수 있습니다.")
        elif best_params.temperature > 0.8:
            recommendations.append("높은 temperature로 창의적이지만 일관성이 떨어질 수 있습니다.")

        if best_params.top_k < 10:
            recommendations.append("낮은 top_k로 결정론적 결과를 생성하지만 다양성이 제한됩니다.")

        # 하드웨어 사용량 기반 추천
        if hardware_usage.get('memory_peak', 0) > 0.9:
            recommendations.append("메모리 사용량이 높습니다. 배치 크기를 줄이거나 양자화를 고려하세요.")

        if hardware_usage.get('gpu_utilization', 0) < 0.5:
            recommendations.append("GPU 활용률이 낮습니다. 배치 크기를 늘려 처리량을 향상시킬 수 있습니다.")

        # 성능 기반 추천
        if optimization_result['best_score'] < 0.7:
            recommendations.append("전체 성능이 낮습니다. 모델 크기를 늘리거나 추가 파인튜닝을 고려하세요.")

        return recommendations

    async def benchmark_model(self, model_name: str, dataset_name: str,
                            params: InferenceParams, num_samples: int = 200,
                            iterations: int = 3) -> BenchmarkResult:
        """모델 벤치마크 실행"""
        test_id = f"bench_{model_name}_{dataset_name}_{int(time.time())}"
        self.logger.info(f"벤치마크 시작: {test_id}")

        # 하드웨어 모니터링 시작
        self.hardware_monitor.start_monitoring()

        # 데이터셋 로드
        samples = await self._load_dataset_for_model(model_name, dataset_name, num_samples)

        # 모델 인터페이스 가져오기
        model_interface = self.model_manager.get_interface(model_name)

        # 평가자 생성
        evaluator = EvaluatorFactory.create_evaluator('performance')

        all_results = []
        performance_metrics_list = []

        for i in range(iterations):
            self.logger.info(f"벤치마크 반복 {i+1}/{iterations}")

            # 프롬프트 생성
            prompts = [self._format_prompt(sample) for sample in samples]

            # 배치 생성 실행
            batch_response = await model_interface.generate_batch_async(prompts, params)

            # 평가 실행
            predictions = [r.content for r in batch_response.responses]
            ground_truths = [s.answer for s in samples[:len(predictions)]]
            generation_stats = [r.generation_stats for r in batch_response.responses]

            evaluation_results = []
            for pred, gt, stats in zip(predictions, ground_truths, generation_stats):
                result = evaluator.evaluate_single(pred, gt, generation_stats=stats)
                evaluation_results.append(result)

            all_results.extend(evaluation_results)

            # 성능 메트릭 계산
            latencies = [r.latency for r in batch_response.responses]
            tokens_per_second = [r.generation_stats.get('tokens_per_second', 0) for r in batch_response.responses]

            perf_metrics = PerformanceMetrics(
                tokens_per_second=np.mean(tokens_per_second),
                memory_usage_mb=batch_response.peak_memory_usage,
                gpu_utilization=0,  # 실제 구현에서는 GPU 모니터링 추가
                cpu_utilization=0,
                latency_p50=np.percentile(latencies, 50),
                latency_p95=np.percentile(latencies, 95),
                latency_p99=np.percentile(latencies, 99),
                throughput=batch_response.throughput
            )

            performance_metrics_list.append(perf_metrics)

        # 하드웨어 모니터링 종료
        hardware_usage = self.hardware_monitor.stop_monitoring()

        # 평균 성능 메트릭 계산
        avg_performance = PerformanceMetrics(
            tokens_per_second=np.mean([p.tokens_per_second for p in performance_metrics_list]),
            memory_usage_mb=np.max([p.memory_usage_mb for p in performance_metrics_list]),
            gpu_utilization=np.mean([p.gpu_utilization for p in performance_metrics_list]),
            cpu_utilization=np.mean([p.cpu_utilization for p in performance_metrics_list]),
            latency_p50=np.mean([p.latency_p50 for p in performance_metrics_list]),
            latency_p95=np.mean([p.latency_p95 for p in performance_metrics_list]),
            latency_p99=np.mean([p.latency_p99 for p in performance_metrics_list]),
            throughput=np.mean([p.throughput for p in performance_metrics_list])
        )

        # 하드웨어 효율성 계산
        hardware_efficiency = self._calculate_hardware_efficiency(avg_performance, hardware_usage)

        # 비용 분석
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

        # 결과 저장
        self._save_benchmark_result(benchmark_result)

        self.logger.info(f"벤치마크 완료: {avg_performance.tokens_per_second:.1f} tokens/sec")

        return benchmark_result

    def _calculate_hardware_efficiency(self, performance: PerformanceMetrics,
                                     hardware_usage: Dict[str, float]) -> Dict[str, float]:
        """하드웨어 효율성 계산"""
        return {
            'tokens_per_watt': performance.tokens_per_second / max(hardware_usage.get('power_consumption', 100), 1),
            'memory_efficiency': performance.tokens_per_second / max(performance.memory_usage_mb, 1),
            'gpu_efficiency': performance.tokens_per_second / max(hardware_usage.get('gpu_utilization', 50), 1),
            'overall_efficiency': performance.tokens_per_second / max(
                hardware_usage.get('power_consumption', 100) *
                performance.memory_usage_mb / 1000, 1
            )
        }

    def _calculate_cost_analysis(self, performance: PerformanceMetrics,
                               hardware_usage: Dict[str, float]) -> Dict[str, float]:
        """비용 분석"""
        # 예시 비용 계산 (실제로는 클라우드 요금이나 전력 비용 기반)
        power_cost_per_hour = 0.1  # $0.1/hour
        memory_cost_per_gb_hour = 0.01  # $0.01/GB/hour

        power_consumption = hardware_usage.get('power_consumption', 100)  # watts
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
                'avg_score': np.mean([r.score for r in result.evaluation_results]),
                'accuracy': np.mean([1 if r.correct else 0 for r in result.evaluation_results])
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

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

        # 별도 스레드에서 모니터링 실행
        import threading
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """모니터링 종료 및 결과 반환"""
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
                # CPU 및 메모리
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()

                metric = {
                    'timestamp': time.time() - self.start_time,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / 1024**3
                }

                # GPU 메트릭 (가능한 경우)
                if torch.cuda.is_available():
                    try:
                        metric['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
                        metric['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
                    except:
                        pass

                self.metrics.append(metric)
                time.sleep(1)  # 1초마다 샘플링

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
            'cpu_avg': np.mean(cpu_values),
            'cpu_peak': np.max(cpu_values),
            'memory_avg': np.mean(memory_values),
            'memory_peak': np.max(memory_values),
            'duration': self.metrics[-1]['timestamp']
        }

        # GPU 메트릭 추가
        gpu_memory_values = [m.get('gpu_memory_allocated', 0) for m in self.metrics]
        if any(gpu_memory_values):
            summary.update({
                'gpu_memory_avg': np.mean(gpu_memory_values),
                'gpu_memory_peak': np.max(gpu_memory_values)
            })

        # 전력 소비 추정 (CPU 사용률 기반)
        estimated_power = 50 + (summary['cpu_avg'] / 100) * 150  # 50W 기본 + CPU 비례
        summary['power_consumption'] = estimated_power

        return summary

    def get_usage_summary(self) -> Dict[str, float]:
        """현재 사용량 요약"""
        return self._calculate_summary()

# 사용 예시
if __name__ == "__main__":
    async def example_optimization():
        print("오픈소스 LLM 성능 최적화 시스템 예시")

        # 최적화기 초기화
        optimizer = PerformanceOptimizer()
        optimizer.setup_models()

        # 사용 가능한 모델이 있는 경우에만 실행
        available_models = optimizer.model_manager.list_models()
        if available_models:
            model_name = available_models[0]

            print(f"모델 최적화 시작: {model_name}")

            # 파라미터 최적화 실행
            result = await optimizer.optimize_inference_params(
                model_name=model_name,
                dataset_name='korean_math',
                optimization_strategy='grid_search',
                max_trials=10,
                num_samples=20
            )

            print(f"최적화 완료!")
            print(f"최고 점수: {result.best_score:.3f}")
            print(f"최적 파라미터:")
            print(f"  Temperature: {result.best_params.temperature}")
            print(f"  Top-p: {result.best_params.top_p}")
            print(f"  Top-k: {result.best_params.top_k}")

            print(f"\n추천사항:")
            for rec in result.recommendations:
                print(f"  - {rec}")

        else:
            print("사용 가능한 모델이 없습니다. 설정을 확인하세요.")

    # 예시 실행
    asyncio.run(example_optimization()) > best_score:
                best_score = score
                best_params = inference_params

            self.logger.info(f"Trial {len(history)}: Score {score:.3f}, Params: {params}")

            return -score  # 최소화 문제로 변환

        # 최적화 실행
        result = gp_minimize(objective, dimensions, n_calls=max_trials, random_state=42)

        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': history
        }

    async def _grid_search_optimization(self, model_interface, samples: List[TestSample],
                                      evaluator, test_id: str) -> Dict[str, Any]:
        """그리드 서치 최적화"""
        param_grid = self.config_manager.get_param_grid()

        # 샘플링으로 조합 수 줄이기
        sampled_combinations = self._sample_param_combinations(param_grid, max_combinations=30)

        history = []
        best_score = 0
        best_params = None

        for i, param_dict in enumerate(sampled_combinations):
            inference_params = InferenceParams(**param_dict)

            self.logger.info(f"그리드 서치 {i+1}/{len(sampled_combinations)}: {param_dict}")

            score = await self._evaluate_params(
                model_interface, samples, evaluator, inference_params
            )

            history.append({
                'params': param_dict,
                'score': score,
                'timestamp': datetime.now().isoformat()
            })

            if score