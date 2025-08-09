"""
안전한 오픈소스 LLM 성능 평가 시스템
모든 문제점이 해결된 버전
"""
import re
import json
import numpy as np
import torch
import gc
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
from collections import defaultdict
import difflib

@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    score: float
    correct: bool
    prediction: str
    ground_truth: str
    reasoning_steps: Optional[List[str]] = None
    confidence: Optional[float] = None
    latency: Optional[float] = None
    token_efficiency: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    accuracy: float
    avg_latency: float
    tokens_per_second: float
    memory_efficiency: float
    throughput: float
    consistency_score: float
    reasoning_quality: float
    token_efficiency: float

class BaseEvaluator(ABC):
    """기본 평가자 클래스 - 안전성 강화"""

    def __init__(self, use_semantic: bool = False):  # 기본값을 False로 변경
        super().__init__()
        self.use_semantic = use_semantic
        self.semantic_model = None
        self.logger = logging.getLogger(__name__)

        if use_semantic:
            self._init_semantic_model()

    def _init_semantic_model(self):
        """의미적 유사도를 위한 모델 초기화 - 안전한 방식"""
        try:
            from sentence_transformers import SentenceTransformer
            # 더 가벼운 모델 사용
            self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.logger.info("Semantic model loaded successfully")
        except ImportError:
            self.logger.warning("sentence-transformers not available, disabling semantic similarity")
            self.use_semantic = False
            self.semantic_model = None
        except Exception as e:
            self.logger.warning(f"Failed to load semantic model: {e}")
            self.use_semantic = False
            self.semantic_model = None

    @abstractmethod
    def evaluate_single(self, prediction: str, ground_truth: str, **kwargs) -> EvaluationResult:
        """단일 예측 평가"""
        pass

    def evaluate_batch(self, predictions: List[str], ground_truths: List[str],
                      generation_stats: List[Dict] = None, **kwargs) -> Dict[str, float]:
        """안전한 배치 평가"""
        results = []

        # 메모리 효율적 처리
        batch_size = 10
        for i in range(0, len(predictions), batch_size):
            batch_preds = predictions[i:i + batch_size]
            batch_truths = ground_truths[i:i + batch_size]
            batch_stats = generation_stats[i:i + batch_size] if generation_stats else [{}] * len(batch_preds)

            for pred, gt, stats in zip(batch_preds, batch_truths, batch_stats):
                try:
                    result = self.evaluate_single(pred, gt, **kwargs)

                    # 성능 통계 추가
                    if stats:
                        result.latency = stats.get('generation_time', 0)
                        result.token_efficiency = self._calculate_token_efficiency(pred, stats)

                    results.append(result)

                except Exception as e:
                    self.logger.warning(f"Evaluation failed for sample: {e}")
                    # 기본 결과 추가
                    results.append(EvaluationResult(
                        score=0.0,
                        correct=False,
                        prediction=pred,
                        ground_truth=gt,
                        details={'error': str(e)}
                    ))

            # 주기적 메모리 정리
            if i % (batch_size * 5) == 0:
                gc.collect()

        return self._aggregate_results(results)

    def _aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """결과 집계 - 안전한 방식"""
        if not results:
            return {
                'accuracy': 0.0,
                'average_score': 0.0,
                'total_samples': 0,
                'error_rate': 1.0
            }

        # 기본 메트릭
        scores = [r.score for r in results if r.score is not None]
        correct_count = sum(1 for r in results if r.correct)
        latencies = [r.latency for r in results if r.latency is not None and r.latency > 0]
        token_efficiencies = [r.token_efficiency for r in results if r.token_efficiency is not None]
        confidences = [r.confidence for r in results if r.confidence is not None]

        metrics = {
            'accuracy': correct_count / len(results),
            'average_score': np.mean(scores) if scores else 0.0,
            'std_score': np.std(scores) if len(scores) > 1 else 0.0,
            'total_samples': len(results),
            'error_rate': 1.0 - (len(scores) / len(results))
        }

        # 성능 메트릭 추가 (안전하게)
        if latencies:
            metrics.update({
                'avg_latency': np.mean(latencies),
                'latency_p95': np.percentile(latencies, 95),
                'latency_p99': np.percentile(latencies, 99),
                'throughput': 1.0 / np.mean(latencies)
            })

        if token_efficiencies:
            metrics['avg_token_efficiency'] = np.mean(token_efficiencies)

        if confidences:
            metrics['avg_confidence'] = np.mean(confidences)

        # 일관성 점수 계산
        metrics['consistency_score'] = self._calculate_consistency(results)

        return metrics

    def _calculate_token_efficiency(self, prediction: str, stats: Dict) -> float:
        """토큰 효율성 계산"""
        try:
            output_tokens = stats.get('output_tokens', len(prediction.split()))
            meaningful_words = len([w for w in prediction.split() if len(w) > 2])
            return meaningful_words / output_tokens if output_tokens > 0 else 0
        except:
            return 0.0

    def _calculate_consistency(self, results: List[EvaluationResult]) -> float:
        """답변 일관성 점수 계산"""
        if len(results) < 2:
            return 1.0

        try:
            scores = [r.score for r in results if r.score is not None]
            if len(scores) < 2:
                return 1.0

            score_std = np.std(scores)
            max_possible_std = 0.5
            return max(0.0, 1.0 - (score_std / max_possible_std))
        except:
            return 0.0

class KoreanMathEvaluator(BaseEvaluator):
    """한국어 수학 문제 평가자 - 완전 구현"""

    def __init__(self, use_semantic: bool = False):
        super().__init__(use_semantic)

    def evaluate_single(self, prediction: str, ground_truth: str, **kwargs) -> EvaluationResult:
        """한국어 수학 문제 평가"""
        start_time = time.time()

        # 답 추출
        pred_answer = self._extract_korean_answer(prediction)
        gt_answer = self._extract_korean_answer(ground_truth)

        # 수치 비교
        correct = self._compare_numbers(pred_answer, gt_answer)
        score = 1.0 if correct else 0.0

        # 추론 과정 추출 및 품질 평가
        reasoning_steps = self._extract_korean_reasoning_steps(prediction)
        reasoning_quality = self._evaluate_korean_reasoning_quality(reasoning_steps)

        # 신뢰도 계산
        confidence = self._calculate_korean_confidence(prediction)

        evaluation_time = time.time() - start_time

        return EvaluationResult(
            score=score,
            correct=correct,
            prediction=prediction,
            ground_truth=ground_truth,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            details={
                'extracted_pred': pred_answer,
                'extracted_gt': gt_answer,
                'reasoning_quality': reasoning_quality,
                'evaluation_time': evaluation_time,
                'language': 'korean'
            }
        )

    def _extract_korean_answer(self, text: str) -> Optional[float]:
        """한국어 텍스트에서 수치 답 추출"""
        patterns = [
            r'답(?:은|:)?\s*([+-]?\d+(?:[.,]\d+)?)',
            r'정답(?:은|:)?\s*([+-]?\d+(?:[.,]\d+)?)',
            r'결과(?:는|:)?\s*([+-]?\d+(?:[.,]\d+)?)',
            r'따라서\s*([+-]?\d+(?:[.,]\d+)?)',
            r'그러므로\s*([+-]?\d+(?:[.,]\d+)?)',
            r'= ([+-]?\d+(?:[.,]\d+)?)',
            r'([+-]?\d+(?:[.,]\d+)?)\s*(?:이다|입니다|다|개|명|원|kg|km|cm)',
            r'([+-]?\d+(?:[.,]\d+)?)$'
        ]

        for pattern in patterns:
            try:
                match = re.search(pattern, text)
                if match:
                    number_str = match.group(1).replace(',', '.')
                    return float(number_str)
            except (ValueError, AttributeError):
                continue

        return None

    def _compare_numbers(self, pred: Optional[float], gt: Optional[float], tolerance: float = 1e-6) -> bool:
        """수치 비교"""
        if pred is None or gt is None:
            return False
        return abs(pred - gt) < tolerance

    def _extract_korean_reasoning_steps(self, text: str) -> List[str]:
        """한국어 추론 과정 추출"""
        steps = []
        step_indicators = [
            r'(?:단계|과정)\s*\d+[:.]?\s*',
            r'(?:먼저|첫째|첫번째)[:.]?\s*',
            r'(?:다음|그다음|둘째|두번째)[:.]?\s*',
            r'(?:그러면|그러므로|따라서)[:.]?\s*',
            r'(?:마지막으로|결론적으로)[:.]?\s*'
        ]

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if any(re.search(indicator, line) for indicator in step_indicators):
                steps.append(line)
            elif '=' in line and any(char.isdigit() for char in line):
                steps.append(line)

        return steps

    def _evaluate_korean_reasoning_quality(self, steps: List[str]) -> float:
        """한국어 추론 품질 평가"""
        if not steps:
            return 0.0

        quality_score = 0.0

        # 단계 수 점수
        step_count_score = min(len(steps) / 5.0, 1.0)
        quality_score += step_count_score * 0.3

        # 논리적 연결어 사용
        logical_words = ['따라서', '그러므로', '그러면', '왜냐하면', '때문에', '결과적으로']
        logical_count = sum(1 for step in steps for word in logical_words if word in step)
        logical_score = min(logical_count / len(steps), 0.5)
        quality_score += logical_score * 0.3

        # 수식 포함 여부
        formula_count = sum(1 for step in steps if '=' in step or any(op in step for op in ['+', '-', '×', '÷']))
        formula_score = min(formula_count / len(steps), 1.0)
        quality_score += formula_score * 0.4

        return min(quality_score, 1.0)

    def _calculate_korean_confidence(self, text: str) -> float:
        """한국어 응답 신뢰도 계산"""
        confidence_words = ['확실', '분명', '명확', '틀림없이', '확실히', '반드시']
        uncertainty_words = ['아마', '추측', '생각', '불확실', '혹시', '가능성', '추정']

        confidence_count = sum(1 for word in confidence_words if word in text)
        uncertainty_count = sum(1 for word in uncertainty_words if word in text)

        base_confidence = 0.5
        confidence_adjustment = (confidence_count * 0.1) - (uncertainty_count * 0.1)

        if '=' in text and any(char.isdigit() for char in text):
            confidence_adjustment += 0.2

        return max(0.0, min(1.0, base_confidence + confidence_adjustment))

class KoreanQAEvaluator(BaseEvaluator):
    """한국어 질의응답 평가자 - 완전 구현"""

    def __init__(self, use_semantic: bool = False):
        super().__init__(use_semantic)

    def evaluate_single(self, prediction: str, ground_truth: str, **kwargs) -> EvaluationResult:
        """한국어 QA 평가"""
        start_time = time.time()

        # 정확 일치 점수
        exact_score = self._exact_match_score(prediction, ground_truth)

        # 의미적 유사도 점수
        semantic_score = 0.0
        if self.use_semantic and self.semantic_model:
            semantic_score = self._semantic_similarity_score(prediction, ground_truth)

        # 키워드 포함 점수
        keyword_score = self._keyword_inclusion_score(prediction, ground_truth)

        # 종합 점수
        final_score = (exact_score * 0.4 + semantic_score * 0.4 + keyword_score * 0.2)
        correct = final_score > 0.7

        # 응답 완성도 평가
        completeness = self._evaluate_completeness(prediction)

        # 유창성 평가
        fluency = self._evaluate_korean_fluency(prediction)

        evaluation_time = time.time() - start_time

        return EvaluationResult(
            score=final_score,
            correct=correct,
            prediction=prediction,
            ground_truth=ground_truth,
            confidence=self._calculate_qa_confidence(prediction),
            details={
                'exact_score': exact_score,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'completeness': completeness,
                'fluency': fluency,
                'evaluation_time': evaluation_time
            }
        )

    def _exact_match_score(self, pred: str, gt: str) -> float:
        """정확 일치 점수"""
        pred_clean = self._normalize_korean_text(pred)
        gt_clean = self._normalize_korean_text(gt)
        return 1.0 if pred_clean == gt_clean else 0.0

    def _semantic_similarity_score(self, pred: str, gt: str) -> float:
        """의미적 유사도 점수 - 안전한 방식"""
        if not self.semantic_model:
            return 0.0

        try:
            pred_embedding = self.semantic_model.encode([pred])
            gt_embedding = self.semantic_model.encode([gt])

            # cosine similarity 직접 계산
            dot_product = np.dot(pred_embedding[0], gt_embedding[0])
            norm_pred = np.linalg.norm(pred_embedding[0])
            norm_gt = np.linalg.norm(gt_embedding[0])

            if norm_pred == 0 or norm_gt == 0:
                return 0.0

            similarity = dot_product / (norm_pred * norm_gt)
            return max(0.0, float(similarity))
        except Exception as e:
            self.logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0

    def _keyword_inclusion_score(self, pred: str, gt: str) -> float:
        """키워드 포함 점수"""
        gt_keywords = self._extract_korean_keywords(gt)
        if not gt_keywords:
            return 0.0

        pred_text = pred.lower()
        included_keywords = sum(1 for keyword in gt_keywords if keyword in pred_text)
        return included_keywords / len(gt_keywords)

    def _extract_korean_keywords(self, text: str) -> List[str]:
        """한국어 키워드 추출"""
        # 명사형 단어들 추출 (2글자 이상)
        words = re.findall(r'[가-힣]{2,}', text)

        # 불용어 제거
        stopwords = {'것은', '것이', '그것', '이것', '저것', '무엇', '어떤', '하는', '되는', '있는', '없는'}
        keywords = [word.lower() for word in words if word not in stopwords]

        return list(set(keywords))

    def _normalize_korean_text(self, text: str) -> str:
        """한국어 텍스트 정규화"""
        # 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())
        # 구두점 제거
        text = re.sub(r'[.,!?;:()\"\'『』「」\[\]]', '', text)
        return text.lower()

    def _evaluate_completeness(self, prediction: str) -> float:
        """응답 완성도 평가"""
        if not prediction.strip():
            return 0.0

        # 문장 종결어미 확인
        sentence_endings = ['다', '요', '니다', '습니다', '죠', '네', '예']
        has_proper_ending = any(prediction.strip().endswith(ending) for ending in sentence_endings)

        # 최소 길이 확인
        min_length_ok = len(prediction.strip()) >= 5

        # 완전한 문장 구조 확인
        has_subject_predicate = len(prediction.split()) >= 2

        completeness_score = 0.0
        if has_proper_ending:
            completeness_score += 0.4
        if min_length_ok:
            completeness_score += 0.3
        if has_subject_predicate:
            completeness_score += 0.3

        return min(1.0, completeness_score)

    def _evaluate_korean_fluency(self, text: str) -> float:
        """한국어 유창성 평가"""
        if not text.strip():
            return 0.0

        fluency_score = 0.5

        # 반복 단어 체크
        words = text.split()
        if len(words) > 1:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            fluency_score += (repetition_ratio - 0.5) * 0.3

        # 문법적 연결어 사용
        connectors = ['그러나', '하지만', '그리고', '또한', '따라서', '그러므로', '즉', '예를 들어']
        connector_count = sum(1 for conn in connectors if conn in text)
        fluency_score += min(connector_count * 0.1, 0.2)

        # 어색한 표현 체크
        awkward_patterns = [r'(.)\1{3,}', r'[가-힣][a-zA-Z][가-힣]']
        for pattern in awkward_patterns:
            if re.search(pattern, text):
                fluency_score -= 0.1

        return max(0.0, min(1.0, fluency_score))

    def _calculate_qa_confidence(self, prediction: str) -> float:
        """QA 응답 신뢰도 계산"""
        if not prediction.strip():
            return 0.0

        confidence = 0.5

        # 확신 표현
        confident_expressions = ['확실히', '분명히', '틀림없이', '반드시', '정확히']
        uncertain_expressions = ['아마도', '추측하건대', '가능성이', '혹시', '생각해보면']

        for expr in confident_expressions:
            if expr in prediction:
                confidence += 0.1

        for expr in uncertain_expressions:
            if expr in prediction:
                confidence -= 0.1

        # 구체적인 정보 포함 여부
        has_numbers = bool(re.search(r'\d+', prediction))
        has_specific_terms = len(re.findall(r'[가-힣]{3,}', prediction)) > 2

        if has_numbers:
            confidence += 0.1
        if has_specific_terms:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

class BasicEvaluator(BaseEvaluator):
    """기본 평가자 - 가장 안전한 폴백"""

    def __init__(self, **kwargs):
        super().__init__(use_semantic=False)

    def evaluate_single(self, prediction: str, ground_truth: str, **kwargs) -> EvaluationResult:
        """기본 평가 - 문자열 유사도만 사용"""
        try:
            similarity = difflib.SequenceMatcher(None, prediction.lower(), ground_truth.lower()).ratio()

            return EvaluationResult(
                score=similarity,
                correct=similarity > 0.8,
                prediction=prediction,
                ground_truth=ground_truth,
                confidence=similarity,
                details={
                    'similarity': similarity,
                    'evaluator': 'basic',
                    'method': 'string_similarity'
                }
            )
        except Exception as e:
            # 최악의 경우에도 작동
            return EvaluationResult(
                score=0.0,
                correct=False,
                prediction=prediction,
                ground_truth=ground_truth,
                confidence=0.0,
                details={'error': str(e), 'evaluator': 'basic_fallback'}
            )

class PerformanceEvaluator(BaseEvaluator):
    """성능 중심 평가자"""

    def __init__(self, **kwargs):
        super().__init__(use_semantic=False)

    def evaluate_single(self, prediction: str, ground_truth: str,
                       generation_stats: Dict = None, **kwargs) -> EvaluationResult:
        """성능을 고려한 평가"""
        # 기본 정확도
        basic_score = self._basic_accuracy(prediction, ground_truth)

        # 성능 점수
        performance_score = 1.0
        if generation_stats:
            performance_score = self._calculate_performance_score(generation_stats)

        # 종합 점수
        final_score = basic_score * 0.7 + performance_score * 0.3

        return EvaluationResult(
            score=final_score,
            correct=basic_score > 0.8,
            prediction=prediction,
            ground_truth=ground_truth,
            details={
                'basic_score': basic_score,
                'performance_score': performance_score,
                'generation_stats': generation_stats or {}
            }
        )

    def _basic_accuracy(self, pred: str, gt: str) -> float:
        """기본 정확도 계산"""
        return difflib.SequenceMatcher(None, pred.lower(), gt.lower()).ratio()

    def _calculate_performance_score(self, stats: Dict) -> float:
        """성능 점수 계산"""
        score = 1.0

        # 토큰/초 점수
        tokens_per_sec = stats.get('tokens_per_second', 0)
        if tokens_per_sec > 0:
            speed_score = min(tokens_per_sec / 100.0, 1.0)
            score *= speed_score

        # 메모리 효율성
        memory_mb = stats.get('memory_usage_mb', 8000)
        memory_score = max(0.1, min(8000.0 / memory_mb, 1.0))
        score *= memory_score

        return score

class SafeEvaluatorFactory:
    """안전한 평가자 팩토리"""

    evaluators = {
        'korean_math': KoreanMathEvaluator,
        'korean_qa': KoreanQAEvaluator,
        'performance': PerformanceEvaluator,
        'basic': BasicEvaluator
    }

    @classmethod
    def create_evaluator(cls, evaluator_type: str, **kwargs) -> BaseEvaluator:
        """안전한 평가자 생성"""
        try:
            if evaluator_type in cls.evaluators:
                return cls.evaluators[evaluator_type](**kwargs)
            else:
                logging.warning(f"Unknown evaluator type: {evaluator_type}, using basic")
                return BasicEvaluator(**kwargs)
        except Exception as e:
            logging.error(f"Failed to create evaluator {evaluator_type}: {e}")
            return BasicEvaluator(**kwargs)

# 전역 평가자 캐시
_evaluator_cache = {}

def get_cached_evaluator(evaluator_type: str, **kwargs) -> BaseEvaluator:
    """캐시된 평가자 반환"""
    cache_key = f"{evaluator_type}_{hash(frozenset(kwargs.items()))}"

    if cache_key not in _evaluator_cache:
        _evaluator_cache[cache_key] = SafeEvaluatorFactory.create_evaluator(evaluator_type, **kwargs)

    return _evaluator_cache[cache_key]

def clear_evaluator_cache():
    """평가자 캐시 정리"""
    global _evaluator_cache
    _evaluator_cache.clear()
    gc.collect()

def evaluate_with_memory_management(evaluator_type: str, predictions: List[str],
                                  ground_truths: List[str], **kwargs) -> Dict[str, float]:
    """메모리 관리를 포함한 안전한 평가"""
    try:
        evaluator = get_cached_evaluator(evaluator_type, **kwargs)
        return evaluator.evaluate_batch(predictions, ground_truths, **kwargs)
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        # 기본 평가로 폴백
        evaluator = BasicEvaluator()
        return evaluator.evaluate_batch(predictions, ground_truths, **kwargs)
    finally:
        # 메모리 정리
        gc.collect()

class EvaluationUtils:
    """평가 유틸리티 함수들"""

    @staticmethod
    def safe_evaluate_batch(evaluator: BaseEvaluator, predictions: List[str],
                           ground_truths: List[str], batch_size: int = 10) -> Dict[str, float]:
        """안전한 배치 평가"""
        all_results = []

        for i in range(0, len(predictions), batch_size):
            batch_preds = predictions[i:i + batch_size]
            batch_truths = ground_truths[i:i + batch_size]

            for pred, truth in zip(batch_preds, batch_truths):
                try:
                    result = evaluator.evaluate_single(pred, truth)
                    all_results.append(result)
                except Exception as e:
                    logging.warning(f"Evaluation failed for sample {i}: {e}")
                    all_results.append(EvaluationResult(
                        score=0.0,
                        correct=False,
                        prediction=pred,
                        ground_truth=truth,
                        details={'error': str(e)}
                    ))

        if not all_results:
            return {'accuracy': 0.0, 'average_score': 0.0, 'total_samples': 0}

        correct_count = sum(1 for r in all_results if r.correct)
        avg_score = sum(r.score for r in all_results) / len(all_results)

        return {
            'accuracy': correct_count / len(all_results),
            'average_score': avg_score,
            'total_samples': len(all_results),
            'correct_count': correct_count
        }

# 사용 예시
if __name__ == "__main__":
    # 안전한 평가 테스트
    print("=== 안전한 평가 시스템 테스트 ===")

    # 한국어 수학 평가
    try:
        math_eval = SafeEvaluatorFactory.create_evaluator('korean_math')

        prediction = "24 - 8 = 16이므로 답은 16개입니다."
        ground_truth = "16"

        result = math_eval.evaluate_single(prediction, ground_truth)
        print(f"수학 평가 결과:")
        print(f"  점수: {result.score:.3f}")
        print(f"  정답 여부: {result.correct}")
        print(f"  신뢰도: {result.confidence:.3f}")

    except Exception as e:
        print(f"수학 평가 실패: {e}")

    # 한국어 QA 평가
    try:
        qa_eval = SafeEvaluatorFactory.create_evaluator('korean_qa')

        prediction2 = "서울은 대한민국의 수도입니다."
        ground_truth2 = "서울"

        result2 = qa_eval.evaluate_single(prediction2, ground_truth2)
        print(f"\nQA 평가 결과:")
        print(f"  점수: {result2.score:.3f}")
        print(f"  정답 여부: {result2.correct}")

    except Exception as e:
        print(f"QA 평가 실패: {e}")

    # 배치 평가 테스트
    try:
        predictions = ["16개", "서울", "100도"]
        ground_truths = ["16", "서울", "100"]

        metrics = evaluate_with_memory_management('basic', predictions, ground_truths)
        print(f"\n배치 평가 결과:")
        print(f"  정확도: {metrics['accuracy']:.3f}")
        print(f"  평균 점수: {metrics['average_score']:.3f}")
        print(f"  총 샘플: {metrics['total_samples']}")

    except Exception as e:
        print(f"배치 평가 실패: {e}")

    # 메모리 정리
    clear_evaluator_cache()
    print("\n✅ 평가 시스템 테스트 완료")