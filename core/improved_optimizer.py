"""
개선된 최적화 시스템
Critical 문제 해결: Optuna로 통일, 스레드 안전성 개선
"""
import asyncio
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path

# Optuna로 통일 (scikit-optimize 제거)
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from .memory_manager import get_resource_manager
from .async_manager import get_async_manager, run_async_safe
from .error_handler import get_global_error_handler, safe_execute, ErrorCategory
from config.model_config import ModelConfig
from config.base_config import ValidationResult


@dataclass
class InferenceParams:
    """추론 파라미터"""
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

    def __post_init__(self):
        """파라미터 검증 및 안전 범위 적용"""
        self.temperature = max(0.0, min(2.0, self.temperature))
        self.top_p = max(0.01, min(1.0, self.top_p))
        self.top_k = max(1, min(100, self.top_k))
        self.max_new_tokens = max(1, min(2048, self.max_new_tokens))
        self.min_new_tokens = max(1, min(self.max_new_tokens, self.min_new_tokens))
        self.repetition_penalty = max(1.0, min(2.0, self.repetition_penalty))
        self.length_penalty = max(0.1, min(2.0, self.length_penalty))
        self.num_beams = max(1, min(8, self.num_beams))

        # 논리 검증
        if self.temperature == 0.0:
            self.do_sample = False

    def validate(self) -> ValidationResult:
        """파라미터 유효성 검증"""
        result = ValidationResult(True)

        if not (0.0 <= self.temperature <= 2.0):
            result.add_error(f"temperature는 0.0-2.0 범위여야 합니다: {self.temperature}")

        if not (0.01 <= self.top_p <= 1.0):
            result.add_error(f"top_p는 0.01-1.0 범위여야 합니다: {self.top_p}")

        if not (1 <= self.top_k <= 100):
            result.add_error(f"top_k는 1-100 범위여야 합니다: {self.top_k}")

        if self.min_new_tokens > self.max_new_tokens:
            result.add_error("min_new_tokens는 max_new_tokens보다 작아야 합니다")

        return result


@dataclass
class OptimizationResult:
    """최적화 결과"""
    trial_id: str
    model_name: str
    dataset_name: str
    best_params: InferenceParams
    best_score: float
    total_trials: int
    successful_trials: int
    optimization_time: float
    study_summary: Dict[str, Any]
    hardware_usage: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime

    def save_to_file(self, filepath: str):
        """파일로 저장"""
        data = {
            'trial_id': self.trial_id,
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'best_params': asdict(self.best_params),
            'best_score': self.best_score,
            'total_trials': self.total_trials,
            'successful_trials': self.successful_trials,
            'optimization_time': self.optimization_time,
            'study_summary': self.study_summary,
            'hardware_usage': self.hardware_usage,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat(),
            'version': '2.0.0'
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class SafeOptimizer:
    """안전한 성능 최적화기"""

    def __init__(self, max_concurrent_trials: int = 1):
        self.max_concurrent_trials = max_concurrent_trials
        self.logger = logging.getLogger(__name__)
        self.resource_manager = get_resource_manager()
        self.async_manager = get_async_manager()
        self.error_handler = get_global_error_handler()

        # 최적화 설정
        self.optimization_timeout = 3600  # 1시간
        self.trial_timeout = 300  # 5분
        self.memory_threshold = 0.85

        # 결과 저장 경로
        self.results_dir = Path("optimization_results")
        self.results_dir.mkdir(exist_ok=True)

        if not OPTUNA_AVAILABLE:
            self.logger.error("Optuna가 설치되지 않았습니다. pip install optuna")
            raise ImportError("Optuna is required for optimization")

    @safe_execute(fallback_result=None)
    async def optimize_parameters(self,
                                  model_name: str,
                                  dataset_name: str,
                                  evaluator_func: Callable,
                                  n_trials: int = 20,
                                  timeout: Optional[int] = None,
                                  study_name: Optional[str] = None) -> OptimizationResult:
        """파라미터 최적화 메인 함수"""
        if timeout is None:
            timeout = self.optimization_timeout

        trial_id = study_name or f"opt_{model_name}_{dataset_name}_{int(time.time())}"

        self.logger.info(f"최적화 시작: {trial_id}")
        start_time = time.time()

        # 메모리 안전성 확인
        if not self.resource_manager.check_memory_safety():
            self.resource_manager.emergency_cleanup_if_needed()

        try:
            # Optuna 스터디 생성
            study = await self._create_study(trial_id)

            # 목적 함수 생성
            objective = self._create_objective_function(
                model_name, dataset_name, evaluator_func
            )

            # 최적화 실행
            best_params, study_summary = await self._run_optimization(
                study, objective, n_trials, timeout
            )

            # 하드웨어 사용량 수집
            hardware_usage = self.resource_manager.get_memory_stats()

            # 추천사항 생성
            recommendations = self._generate_recommendations(
                best_params, study_summary, hardware_usage
            )

            # 결과 생성
            result = OptimizationResult(
                trial_id=trial_id,
                model_name=model_name,
                dataset_name=dataset_name,
                best_params=best_params,
                best_score=study.best_value,
                total_trials=len(study.trials),
                successful_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                optimization_time=time.time() - start_time,
                study_summary=study_summary,
                hardware_usage=self._extract_hardware_metrics(hardware_usage),
                recommendations=recommendations,
                timestamp=datetime.now()
            )

            # 결과 저장
            result_path = self.results_dir / f"{trial_id}.json"
            result.save_to_file(str(result_path))

            self.logger.info(f"최적화 완료: {trial_id}, 최고 점수: {result.best_score:.4f}")
            return result

        except Exception as e:
            error_info = self.error_handler.handle_exception(
                e,
                context={
                    'operation': 'parameter_optimization',
                    'model': model_name,
                    'dataset': dataset_name
                }
            )
            self.logger.error(f"최적화 실패: {trial_id}")
            raise
        finally:
            # 메모리 정리
            self.resource_manager.cleanup_all_devices()

    async def _create_study(self, study_name: str) -> optuna.Study:
        """Optuna 스터디 생성"""
        # 샘플러 선택 (TPE 권장)
        sampler = TPESampler(
            seed=42,
            n_startup_trials=5,
            n_ei_candidates=24
        )

        # 프루너 선택 (조기 종료)
        pruner = MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2
        )

        # 스터디 생성
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )

        return study

    def _create_objective_function(self,
                                   model_name: str,
                                   dataset_name: str,
                                   evaluator_func: Callable) -> Callable:
        """목적 함수 생성"""

        def objective(trial: optuna.Trial) -> float:
            try:
                # 파라미터 제안
                params = self._suggest_parameters(trial)

                # 메모리 확인
                if not self.resource_manager.check_memory_safety():
                    raise optuna.TrialPruned("메모리 부족으로 trial 중단")

                # 평가 실행
                score = run_async_safe(
                    self._evaluate_parameters_async(
                        model_name, dataset_name, params, evaluator_func
                    ),
                    timeout=self.trial_timeout
                )

                if score is None:
                    raise optuna.TrialPruned("평가 실패로 trial 중단")

                # 중간 결과 리포트 (프루닝용)
                trial.report(score, step=0)

                # 프루닝 확인
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            except Exception as e:
                self.logger.warning(f"Trial {trial.number} 실패: {e}")
                # 에러는 Optuna가 처리하도록 재발생
                raise

        return objective

    def _suggest_parameters(self, trial: optuna.Trial) -> InferenceParams:
        """파라미터 제안"""
        # 안전한 범위 내에서 파라미터 제안
        temperature = trial.suggest_float('temperature', 0.0, 1.0, step=0.1)
        top_p = trial.suggest_float('top_p', 0.1, 1.0, step=0.1)
        top_k = trial.suggest_int('top_k', 1, 100, step=10)
        max_new_tokens = trial.suggest_int('max_new_tokens', 50, 512, step=50)
        repetition_penalty = trial.suggest_float('repetition_penalty', 1.0, 1.5, step=0.1)

        return InferenceParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty
        )

    async def _evaluate_parameters_async(self,
                                         model_name: str,
                                         dataset_name: str,
                                         params: InferenceParams,
                                         evaluator_func: Callable) -> Optional[float]:
        """비동기 파라미터 평가"""
        try:
            # 실제 평가 로직은 evaluator_func에 위임
            score = await evaluator_func(model_name, dataset_name, params)

            # 스코어 검증
            if not isinstance(score, (int, float)) or score < 0:
                self.logger.warning(f"잘못된 스코어: {score}")
                return None

            return float(score)

        except Exception as e:
            self.error_handler.handle_exception(
                e,
                context={
                    'operation': 'parameter_evaluation',
                    'model': model_name,
                    'params': asdict(params)
                }
            )
            return None

    async def _run_optimization(self,
                                study: optuna.Study,
                                objective: Callable,
                                n_trials: int,
                                timeout: int) -> tuple:
        """최적화 실행"""
        start_time = time.time()

        # 타임아웃과 함께 최적화 실행
        def run_study():
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=1,  # 스레드 안전성을 위해 단일 작업
                show_progress_bar=False  # 로깅과 충돌 방지
            )

        # 스레드풀에서 실행
        await self.async_manager.submit_func(
            run_study,
            executor_type=self.async_manager.ExecutorType.THREAD,
            name=f"optuna_study_{study.study_name}"
        )

        # 최적 파라미터 추출
        if study.best_trial is None:
            raise ValueError("유효한 trial이 없습니다")

        best_params = InferenceParams(**study.best_trial.params)

        # 스터디 요약
        study_summary = {
            'best_value': study.best_value,
            'best_trial_number': study.best_trial.number,
            'n_trials': len(study.trials),
            'optimization_time': time.time() - start_time,
            'successful_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        }

        return best_params, study_summary

    def _extract_hardware_metrics(self, memory_stats: Dict) -> Dict[str, float]:
        """하드웨어 메트릭 추출"""
        metrics = {}

        for device, stats in memory_stats.items():
            if hasattr(stats, 'utilization'):
                metrics[f'{device}_utilization'] = stats.utilization
                metrics[f'{device}_allocated_gb'] = stats.allocated_gb
                metrics[f'{device}_total_gb'] = stats.total_gb

        return metrics

    def _generate_recommendations(self,
                                  best_params: InferenceParams,
                                  study_summary: Dict[str, Any],
                                  hardware_usage: Dict) -> List[str]:
        """추천사항 생성"""
        recommendations = []

        # 파라미터 기반 추천
        if best_params.temperature < 0.1:
            recommendations.append(
                "매우 낮은 temperature로 일관성이 높지만 창의성이 제한될 수 있습니다."
            )
        elif best_params.temperature > 0.8:
            recommendations.append(
                "높은 temperature로 창의적이지만 일관성이 떨어질 수 있습니다."
            )

        if best_params.top_p < 0.3:
            recommendations.append(
                "낮은 top_p로 보수적인 토큰 선택을 합니다."
            )

        # 최적화 결과 기반 추천
        success_rate = study_summary['successful_trials'] / study_summary['n_trials']
        if success_rate < 0.5:
            recommendations.append(
                f"Trial 성공률이 낮습니다 ({success_rate:.1%}). 메모리나 시간 제한을 확인하세요."
            )

        # 하드웨어 기반 추천
        for device, utilization in hardware_usage.items():
            if 'utilization' in device and utilization > 0.9:
                recommendations.append(
                    f"{device.replace('_utilization', '')} 사용률이 높습니다 ({utilization:.1%}). "
                    "배치 크기를 줄이거나 양자화를 고려하세요."
                )

        # 성능 기반 추천
        if study_summary['best_value'] > 0.9:
            recommendations.append(
                "매우 높은 성능입니다. 이 설정을 프로덕션에 적용하는 것을 권장합니다."
            )
        elif study_summary['best_value'] < 0.7:
            recommendations.append(
                "성능이 낮습니다. 더 큰 모델이나 다른 접근법을 고려하세요."
            )

        return recommendations

    async def grid_search_optimization(self,
                                       model_name: str,
                                       dataset_name: str,
                                       evaluator_func: Callable,
                                       param_grid: Optional[Dict] = None) -> OptimizationResult:
        """그리드 서치 최적화 (폴백용)"""
        if param_grid is None:
            param_grid = {
                'temperature': [0.0, 0.1, 0.3, 0.7],
                'top_p': [0.3, 0.7, 0.9],
                'top_k': [10, 30, 50],
                'max_new_tokens': [100, 200, 300],
                'repetition_penalty': [1.0, 1.1]
            }

        trial_id = f"grid_{model_name}_{dataset_name}_{int(time.time())}"
        self.logger.info(f"그리드 서치 시작: {trial_id}")

        start_time = time.time()
        best_score = 0.0
        best_params = None
        trial_results = []

        # 조합 생성
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        combinations = list(itertools.product(*param_values))

        # 최대 20개로 제한 (안전성)
        if len(combinations) > 20:
            import random
            combinations = random.sample(combinations, 20)

        # 각 조합 평가
        for i, combination in enumerate(combinations):
            param_dict = dict(zip(param_names, combination))
            params = InferenceParams(**param_dict)

            self.logger.info(f"그리드 서치 {i + 1}/{len(combinations)}: {param_dict}")

            try:
                score = await self._evaluate_parameters_async(
                    model_name, dataset_name, params, evaluator_func
                )

                if score is not None and score > best_score:
                    best_score = score
                    best_params = params

                trial_results.append({
                    'trial': i,
                    'params': param_dict,
                    'score': score,
                    'timestamp': time.time()
                })

            except Exception as e:
                self.logger.warning(f"그리드 서치 trial {i} 실패: {e}")
                trial_results.append({
                    'trial': i,
                    'params': param_dict,
                    'score': None,
                    'error': str(e),
                    'timestamp': time.time()
                })

        if best_params is None:
            # 기본값 사용
            best_params = InferenceParams()
            best_score = 0.0

        # 결과 생성
        hardware_usage = self.resource_manager.get_memory_stats()

        result = OptimizationResult(
            trial_id=trial_id,
            model_name=model_name,
            dataset_name=dataset_name,
            best_params=best_params,
            best_score=best_score,
            total_trials=len(combinations),
            successful_trials=len([r for r in trial_results if r['score'] is not None]),
            optimization_time=time.time() - start_time,
            study_summary={
                'method': 'grid_search',
                'param_grid': param_grid,
                'trial_results': trial_results[:10]  # 처음 10개만 저장
            },
            hardware_usage=self._extract_hardware_metrics(hardware_usage),
            recommendations=self._generate_recommendations(
                best_params, {'best_value': best_score, 'n_trials': len(combinations),
                              'successful_trials': len([r for r in trial_results if r['score'] is not None])},
                self._extract_hardware_metrics(hardware_usage)
            ),
            timestamp=datetime.now()
        )

        # 결과 저장
        result_path = self.results_dir / f"{trial_id}.json"
        result.save_to_file(str(result_path))

        self.logger.info(f"그리드 서치 완료: {trial_id}, 최고 점수: {best_score:.4f}")
        return result

    def load_optimization_result(self, trial_id: str) -> Optional[OptimizationResult]:
        """저장된 최적화 결과 로드"""
        result_path = self.results_dir / f"{trial_id}.json"

        if not result_path.exists():
            return None

        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # InferenceParams 복원
            best_params = InferenceParams(**data['best_params'])

            return OptimizationResult(
                trial_id=data['trial_id'],
                model_name=data['model_name'],
                dataset_name=data['dataset_name'],
                best_params=best_params,
                best_score=data['best_score'],
                total_trials=data['total_trials'],
                successful_trials=data['successful_trials'],
                optimization_time=data['optimization_time'],
                study_summary=data['study_summary'],
                hardware_usage=data['hardware_usage'],
                recommendations=data['recommendations'],
                timestamp=datetime.fromisoformat(data['timestamp'])
            )

        except Exception as e:
            self.logger.error(f"최적화 결과 로드 실패: {trial_id}, {e}")
            return None

    def list_optimization_results(self) -> List[str]:
        """저장된 최적화 결과 목록"""
        return [f.stem for f in self.results_dir.glob("*.json")]

    def cleanup_old_results(self, max_age_days: int = 30):
        """오래된 결과 정리"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600

        cleaned_count = 0
        for result_file in self.results_dir.glob("*.json"):
            if current_time - result_file.stat().st_mtime > max_age_seconds:
                result_file.unlink()
                cleaned_count += 1

        if cleaned_count > 0:
            self.logger.info(f"오래된 최적화 결과 {cleaned_count}개 정리됨")


# 편의 함수들
async def quick_optimize(model_name: str,
                         dataset_name: str,
                         evaluator_func: Callable,
                         n_trials: int = 10) -> OptimizationResult:
    """빠른 최적화"""
    optimizer = SafeOptimizer()
    return await optimizer.optimize_parameters(
        model_name, dataset_name, evaluator_func, n_trials
    )


async def safe_grid_search(model_name: str,
                           dataset_name: str,
                           evaluator_func: Callable,
                           param_grid: Optional[Dict] = None) -> OptimizationResult:
    """안전한 그리드 서치"""
    optimizer = SafeOptimizer()
    return await optimizer.grid_search_optimization(
        model_name, dataset_name, evaluator_func, param_grid
    )


# 사용 예시
if __name__ == "__main__":
    async def dummy_evaluator(model_name: str, dataset_name: str, params: InferenceParams) -> float:
        """더미 평가 함수"""
        import random
        await asyncio.sleep(0.1)  # 평가 시간 시뮬레이션

        # 파라미터에 따른 가상 점수
        score = 0.5 + (1.0 - params.temperature) * 0.3 + params.top_p * 0.2
        score += random.uniform(-0.1, 0.1)  # 노이즈
        return max(0.0, min(1.0, score))


    async def test_optimizer():
        print("=== 개선된 최적화 시스템 테스트 ===")

        if not OPTUNA_AVAILABLE:
            print("❌ Optuna가 설치되지 않았습니다.")
            return

        optimizer = SafeOptimizer()

        try:
            # 1. Optuna 최적화 테스트
            print("1. Optuna 최적화 테스트...")
            result = await optimizer.optimize_parameters(
                model_name="test-model",
                dataset_name="test-dataset",
                evaluator_func=dummy_evaluator,
                n_trials=5,
                timeout=60
            )

            print(f"   최고 점수: {result.best_score:.4f}")
            print(f"   최적 파라미터: temperature={result.best_params.temperature:.3f}")
            print(f"   총 시행: {result.total_trials}")
            print(f"   성공 시행: {result.successful_trials}")
            print(f"   추천사항: {len(result.recommendations)}개")

            # 2. 그리드 서치 테스트
            print("\n2. 그리드 서치 테스트...")
            grid_result = await optimizer.grid_search_optimization(
                model_name="test-model",
                dataset_name="test-dataset",
                evaluator_func=dummy_evaluator,
                param_grid={
                    'temperature': [0.1, 0.3, 0.7],
                    'top_p': [0.7, 0.9],
                    'top_k': [30, 50]
                }
            )

            print(f"   그리드 서치 최고 점수: {grid_result.best_score:.4f}")
            print(f"   소요 시간: {grid_result.optimization_time:.1f}초")

            # 3. 결과 로드 테스트
            print("\n3. 결과 저장/로드 테스트...")
            loaded_result = optimizer.load_optimization_result(result.trial_id)
            if loaded_result:
                print(f"   로드된 결과: {loaded_result.trial_id}")
                print(f"   점수 일치: {loaded_result.best_score == result.best_score}")

            # 4. 결과 목록 조회
            results_list = optimizer.list_optimization_results()
            print(f"   저장된 결과 수: {len(results_list)}")

        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()

        print("\n✅ 개선된 최적화 시스템 테스트 완료")


    # 테스트 실행
    asyncio.run(test_optimizer())