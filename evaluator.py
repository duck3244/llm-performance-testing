"""
오픈소스 LLM 성능 평가를 위한 메트릭 및 채점 시스템
"""
import re
import json
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
from collections import defaultdict

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
    """기본 평가자 클래스"""

    def __init__(self, use_semantic: bool = True):
        super().__init__()
        self.use_semantic = use_semantic
        if use_semantic:
            self._init_semantic_model()

    def _init_semantic_model(self):
        """의미적 유사도를 위한 모델 초기화"""
        try:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        except ImportError:
            self.use_semantic = False
            self.semantic_model = None

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

        # 종합 점수 (가중 평균)
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
        """의미적 유사도 점수"""
        if not self.semantic_model:
            return 0.0

        try:
            pred_embedding = self.semantic_model.encode([pred])
            gt_embedding = self.semantic_model.encode([gt])

            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(pred_embedding, gt_embedding)[0][0]
            return max(0.0, similarity)
        except Exception:
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
        # 간단한 키워드 추출 (실제로는 형태소 분석기 사용 권장)
        import re

        # 명사형 단어들 추출 (2글자 이상)
        words = re.findall(r'[가-힣]{2,}', text)

        # 불용어 제거
        stopwords = {'것은', '것이', '그것', '이것', '저것', '무엇', '어떤', '하는', '되는', '있는', '없는'}
        keywords = [word.lower() for word in words if word not in stopwords]

        return list(set(keywords))

    def _normalize_korean_text(self, text: str) -> str:
        """한국어 텍스트 정규화"""
        import re

        # 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())

        # 구두점 제거
        text = re.sub(r'[.,!?;:()\"\'『』「」\[\]]', '', text)

        # 소문자 변환 (영어 부분)
        return text.lower()

    def _evaluate_completeness(self, prediction: str) -> float:
        """응답 완성도 평가"""
        # 문장 완성도 체크
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

        fluency_score = 0.5  # 기본 점수

        # 반복 단어 체크 (감점)
        words = text.split()
        if len(words) > 1:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            fluency_score += (repetition_ratio - 0.5) * 0.3

        # 문법적 연결어 사용 (가점)
        connectors = ['그러나', '하지만', '그리고', '또한', '따라서', '그러므로', '즉', '예를 들어']
        connector_count = sum(1 for conn in connectors if conn in text)
        fluency_score += min(connector_count * 0.1, 0.2)

        # 어색한 표현 체크 (감점)
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

class MultilingualEvaluator(BaseEvaluator):
    """다국어 지원 평가자"""

    def __init__(self, primary_language: str = "ko"):
        super().__init__(primary_language)
        self.language_evaluators = {
            'ko': KoreanQAEvaluator(),
            'en': EnglishEvaluator(),
            'ja': JapaneseEvaluator(),
            'zh': ChineseEvaluator()
        }

    def evaluate_single(self, prediction: str, ground_truth: str, **kwargs) -> EvaluationResult:
        """언어를 감지하고 적절한 평가자 사용"""
        detected_lang = self._detect_language(prediction)

        if detected_lang in self.language_evaluators:
            evaluator = self.language_evaluators[detected_lang]
            return evaluator.evaluate_single(prediction, ground_truth, **kwargs)
        else:
            # 기본 평가자 사용
            return self._default_evaluate(prediction, ground_truth, **kwargs)

    def _detect_language(self, text: str) -> str:
        """간단한 언어 감지"""
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        japanese_chars = len(re.findall(r'[ひらがなカタカナ]', text))
        chinese_chars = len(re.findall(r'[一-龯]', text))

        total_chars = korean_chars + english_chars + japanese_chars + chinese_chars

        if total_chars == 0:
            return self.language

        ratios = {
            'ko': korean_chars / total_chars,
            'en': english_chars / total_chars,
            'ja': japanese_chars / total_chars,
            'zh': chinese_chars / total_chars
        }

        return max(ratios, key=ratios.get)

    def _default_evaluate(self, prediction: str, ground_truth: str, **kwargs) -> EvaluationResult:
        """기본 평가 로직"""
        similarity = self._calculate_string_similarity(prediction, ground_truth)
        return EvaluationResult(
            score=similarity,
            correct=similarity > 0.8,
            prediction=prediction,
            ground_truth=ground_truth,
            details={'language': 'unknown', 'similarity': similarity}
        )

    def _calculate_string_similarity(self, s1: str, s2: str) -> float:
        """문자열 유사도 계산"""
        import difflib
        return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

# 추가 언어별 평가자 (간단한 구현)
class EnglishEvaluator(BaseEvaluator):
    """영어 평가자"""

    def evaluate_single(self, prediction: str, ground_truth: str, **kwargs) -> EvaluationResult:
        import difflib
        similarity = difflib.SequenceMatcher(None, prediction.lower(), ground_truth.lower()).ratio()
        return EvaluationResult(
            score=similarity,
            correct=similarity > 0.8,
            prediction=prediction,
            ground_truth=ground_truth,
            details={'language': 'english'}
        )

class JapaneseEvaluator(BaseEvaluator):
    """일본어 평가자"""

    def evaluate_single(self, prediction: str, ground_truth: str, **kwargs) -> EvaluationResult:
        import difflib
        similarity = difflib.SequenceMatcher(None, prediction, ground_truth).ratio()
        return EvaluationResult(
            score=similarity,
            correct=similarity > 0.8,
            prediction=prediction,
            ground_truth=ground_truth,
            details={'language': 'japanese'}
        )

class ChineseEvaluator(BaseEvaluator):
    """중국어 평가자"""

    def evaluate_single(self, prediction: str, ground_truth: str, **kwargs) -> EvaluationResult:
        import difflib
        similarity = difflib.SequenceMatcher(None, prediction, ground_truth).ratio()
        return EvaluationResult(
            score=similarity,
            correct=similarity > 0.8,
            prediction=prediction,
            ground_truth=ground_truth,
            details={'language': 'chinese'}
        )

class PerformanceEvaluator(BaseEvaluator):
    """성능 중심 평가자"""

    def evaluate_single(self, prediction: str, ground_truth: str,
                       generation_stats: Dict = None, **kwargs) -> EvaluationResult:
        """성능을 고려한 평가"""
        # 기본 정확도
        basic_score = self._basic_accuracy(prediction, ground_truth)

        # 성능 점수 (속도, 효율성)
        performance_score = 1.0
        if generation_stats:
            performance_score = self._calculate_performance_score(generation_stats)

        # 종합 점수 (정확도 70% + 성능 30%)
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
        import difflib
        return difflib.SequenceMatcher(None, pred.lower(), gt.lower()).ratio()

    def _calculate_performance_score(self, stats: Dict) -> float:
        """성능 점수 계산"""
        score = 1.0

        # 토큰/초 점수 (100 토큰/초를 기준점으로)
        tokens_per_sec = stats.get('tokens_per_second', 0)
        if tokens_per_sec > 0:
            speed_score = min(tokens_per_sec / 100.0, 1.0)
            score *= speed_score

        # 메모리 효율성 (8GB를 기준점으로)
        memory_mb = stats.get('memory_usage_mb', 8000)
        memory_score = max(0.1, min(8000.0 / memory_mb, 1.0))
        score *= memory_score

        return score

class EvaluatorFactory:
    """평가자 팩토리"""

    evaluators = {
        'korean_math': KoreanMathEvaluator,
        'korean_qa': KoreanQAEvaluator,
        'multilingual': MultilingualEvaluator,
        'performance': PerformanceEvaluator,
        'english': EnglishEvaluator,
        'japanese': JapaneseEvaluator,
        'chinese': ChineseEvaluator
    }

    @classmethod
    def create_evaluator(cls, evaluator_type: str, **kwargs) -> BaseEvaluator:
        """평가자 생성"""
        if evaluator_type not in cls.evaluators:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")

        return cls.evaluators[evaluator_type](**kwargs)

    @classmethod
    def list_evaluators(cls) -> List[str]:
        """사용 가능한 평가자 목록"""
        return list(cls.evaluators.keys())

# 사용 예시
if __name__ == "__main__":
    # 한국어 수학 평가 테스트
    math_eval = EvaluatorFactory.create_evaluator('korean_math')

    prediction = """
    이 문제를 단계별로 풀어보겠습니다.
    단계 1: 전체 사탕 개수는 24개입니다.
    단계 2: 친구들에게 준 사탕은 8개입니다.
    단계 3: 24 - 8 = 16
    따라서 제임스에게 남은 사탕은 16개입니다.
    """
    ground_truth = "16"

    result = math_eval.evaluate_single(prediction, ground_truth)
    print(f"수학 평가 결과:")
    print(f"점수: {result.score:.3f}")
    print(f"정답 여부: {result.correct}")
    print(f"추론 단계 수: {len(result.reasoning_steps or [])}")
    print(f"신뢰도: {result.confidence:.3f}")

    # 한국어 QA 평가 테스트
    qa_eval = EvaluatorFactory.create_evaluator('korean_qa')

    prediction2 = "서울은 대한민국의 수도이며, 약 1000만 명의 인구가 거주하는 대도시입니다."
    ground_truth2 = "서울은 한국의 수도입니다."

    result2 = qa_eval.evaluate_single(prediction2, ground_truth2)
    print(f"\nQA 평가 결과:")
    print(f"점수: {result2.score:.3f}")
    print(f"정답 여부: {result2.correct}")
    if result2.details:
        print(f"의미적 유사도: {result2.details.get('semantic_score', 0):.3f}")
        print(f"완성도: {result2.details.get('completeness', 0):.3f}")

    # 성능 평가 테스트
    perf_eval = EvaluatorFactory.create_evaluator('performance')

    generation_stats = {
        'tokens_per_second': 75.5,
        'memory_usage_mb': 6400,
        'generation_time': 0.8
    }

    result3 = perf_eval.evaluate_single(
        prediction2, ground_truth2,
        generation_stats=generation_stats
    )
    print(f"\n성능 평가 결과:")
    print(f"종합 점수: {result3.score:.3f}")
    if result3.details:
        print(f"성능 점수: {result3.details.get('performance_score', 0):.3f}")
        print(f"기본 점수: {result3.details.get('basic_score', 0):.3f}")
self, language: str = "ko"):
        self.language = language

    @abstractmethod
    def evaluate_single(self, prediction: str, ground_truth: str, **kwargs) -> EvaluationResult:
        """단일 예측 평가"""
        pass

    def evaluate_batch(self, predictions: List[str], ground_truths: List[str],
                      generation_stats: List[Dict] = None, **kwargs) -> Dict[str, float]:
        """배치 평가"""
        results = []

        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            stats = generation_stats[i] if generation_stats else {}

            result = self.evaluate_single(pred, gt, **kwargs)

            # 성능 통계 추가
            if stats:
                result.latency = stats.get('generation_time', 0)
                result.token_efficiency = self._calculate_token_efficiency(pred, stats)

            results.append(result)

        return self._aggregate_results(results)

    def _aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """결과 집계"""
        if not results:
            return {}

        # 기본 메트릭
        scores = [r.score for r in results]
        correct_count = sum(1 for r in results if r.correct)
        latencies = [r.latency for r in results if r.latency is not None]
        token_efficiencies = [r.token_efficiency for r in results if r.token_efficiency is not None]
        confidences = [r.confidence for r in results if r.confidence is not None]

        metrics = {
            'accuracy': correct_count / len(results),
            'average_score': np.mean(scores),
            'std_score': np.std(scores),
            'total_samples': len(results)
        }

        # 성능 메트릭 추가
        if latencies:
            metrics.update({
                'avg_latency': np.mean(latencies),
                'latency_p95': np.percentile(latencies, 95),
                'latency_p99': np.percentile(latencies, 99),
                'throughput': 1.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0
            })

        if token_efficiencies:
            metrics['avg_token_efficiency'] = np.mean(token_efficiencies)

        if confidences:
            metrics['avg_confidence'] = np.mean(confidences)
            metrics['confidence_calibration'] = self._calculate_calibration(results)

        # 일관성 점수 계산
        metrics['consistency_score'] = self._calculate_consistency(results)

        return metrics

    def _calculate_token_efficiency(self, prediction: str, stats: Dict) -> float:
        """토큰 효율성 계산 (의미있는 정보 / 사용된 토큰 수)"""
        output_tokens = stats.get('output_tokens', len(prediction.split()))
        meaningful_words = len([w for w in prediction.split() if len(w) > 2])
        return meaningful_words / output_tokens if output_tokens > 0 else 0

    def _calculate_calibration(self, results: List[EvaluationResult]) -> float:
        """신뢰도 보정 점수 계산"""
        if not any(r.confidence is not None for r in results):
            return 0.0

        # 신뢰도와 실제 정확도 간의 상관관계
        confidence_bins = defaultdict(list)
        for result in results:
            if result.confidence is not None:
                bin_idx = int(result.confidence * 10)  # 0.1 단위로 분할
                confidence_bins[bin_idx].append(1 if result.correct else 0)

        calibration_error = 0.0
        total_samples = 0

        for bin_idx, accuracies in confidence_bins.items():
            if accuracies:
                bin_confidence = bin_idx / 10.0
                bin_accuracy = np.mean(accuracies)
                bin_size = len(accuracies)
                calibration_error += bin_size * abs(bin_confidence - bin_accuracy)
                total_samples += bin_size

        return 1.0 - (calibration_error / total_samples) if total_samples > 0 else 0.0

    def _calculate_consistency(self, results: List[EvaluationResult]) -> float:
        """답변 일관성 점수 계산"""
        if len(results) < 2:
            return 1.0

        # 점수의 표준편차를 이용한 일관성 측정
        scores = [r.score for r in results]
        score_std = np.std(scores)
        max_possible_std = 0.5  # 최대 가능한 표준편차

        return max(0.0, 1.0 - (score_std / max_possible_std))

class KoreanMathEvaluator(BaseEvaluator):
    """한국어 수학 문제 평가자"""

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
        # 한국어 패턴들
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
            match = re.search(pattern, text)
            if match:
                try:
                    # 쉼표를 점으로 변환
                    number_str = match.group(1).replace(',', '.')
                    return float(number_str)
                except ValueError:
                    continue

        return None

    def _compare_numbers(self, pred: Optional[float], gt: Optional[float], tolerance: float = 1e-6) -> bool:
        """수치 비교 (한국어 특수 케이스 고려)"""
        if pred is None or gt is None:
            return False
        return abs(pred - gt) < tolerance

    def _extract_korean_reasoning_steps(self, text: str) -> List[str]:
        """한국어 추론 과정 추출"""
        steps = []

        # 한국어 단계 지시어들
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

        # 단계 수 점수 (1-5단계가 적절)
        step_count_score = min(len(steps) / 5.0, 1.0)
        quality_score += step_count_score * 0.3

        # 논리적 연결어 사용
        logical_words = ['따라서', '그러므로', '그러면', '왜냐하면', '때문에', '결과적으로']
        logical_count = sum(1 for step in steps for word in logical_words if word in step)
        logical_score = min(logical_count / len(steps), 0.5)
        quality_score += logical_score * 0.3

        # 수식 포함 여부
        formula_count = sum(1 for step in steps if '=' in step or '+' in step or '-' in step or '×' in step or '÷' in step)
        formula_score = min(formula_count / len(steps), 1.0)
        quality_score += formula_score * 0.4

        return min(quality_score, 1.0)

    def _calculate_korean_confidence(self, text: str) -> float:
        """한국어 응답 신뢰도 계산"""
        confidence_words = ['확실', '분명', '명확', '틀림없이', '확실히', '반드시']
        uncertainty_words = ['아마', '추측', '생각', '불확실', '혹시', '가능성', '추정']

        confidence_count = sum(1 for word in confidence_words if word in text)
        uncertainty_count = sum(1 for word in uncertainty_words if word in text)

        # 기본 신뢰도 0.5에서 시작
        base_confidence = 0.5
        confidence_adjustment = (confidence_count * 0.1) - (uncertainty_count * 0.1)

        # 수식이나 명확한 계산 과정이 있으면 신뢰도 증가
        if '=' in text and any(char.isdigit() for char in text):
            confidence_adjustment += 0.2

        return max(0.0, min(1.0, base_confidence + confidence_adjustment))

class KoreanQAEvaluator(BaseEvaluator):
    """한국어 질의응답 평가자"""

    def __init__(