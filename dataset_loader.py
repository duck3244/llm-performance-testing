"""
오픈소스 LLM 테스트를 위한 다국어 데이터셋 로딩 및 관리
"""
import json
import csv
import random
import requests
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

@dataclass
class TestSample:
    """테스트 샘플 데이터 클래스"""
    id: str
    question: str
    answer: str
    context: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseDatasetLoader(ABC):
    """기본 데이터셋 로더"""

    def __init__(self, dataset_path: str, num_samples: Optional[int] = None,
                 random_seed: int = 42, language: str = "ko"):
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.language = language
        random.seed(random_seed)

    @abstractmethod
    def load_samples(self) -> List[TestSample]:
        """샘플 로드"""
        pass

    def get_iterator(self, batch_size: int = 1) -> Iterator[List[TestSample]]:
        """배치 단위 이터레이터"""
        samples = self.load_samples()
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def get_sample_by_id(self, sample_id: str) -> Optional[TestSample]:
        """ID로 샘플 검색"""
        samples = self.load_samples()
        for sample in samples:
            if sample.id == sample_id:
                return sample
        return None

    def filter_by_language(self, language: str) -> List[TestSample]:
        """언어별 필터링"""
        samples = self.load_samples()
        return [s for s in samples if s.language == language]

    def filter_by_category(self, category: str) -> List[TestSample]:
        """카테고리별 필터링"""
        samples = self.load_samples()
        return [s for s in samples if s.category == category]

    def filter_by_difficulty(self, difficulty: str) -> List[TestSample]:
        """난이도별 필터링"""
        samples = self.load_samples()
        return [s for s in samples if s.difficulty == difficulty]

class JSONDatasetLoader(BaseDatasetLoader):
    """JSON 형식 데이터셋 로더"""

    def __init__(self, dataset_path: str, question_key: str = "question",
                 answer_key: str = "answer", **kwargs):
        super().__init__(dataset_path, **kwargs)
        self.question_key = question_key
        self.answer_key = answer_key

    def load_samples(self) -> List[TestSample]:
        """JSON 파일에서 샘플 로드"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                sample = self._create_sample_from_dict(str(i), item)
                if sample:
                    samples.append(sample)
        elif isinstance(data, dict):
            for key, item in data.items():
                sample = self._create_sample_from_dict(key, item)
                if sample:
                    samples.append(sample)

        if self.num_samples and len(samples) > self.num_samples:
            samples = random.sample(samples, self.num_samples)

        return samples

    def _create_sample_from_dict(self, sample_id: str, data: Dict[str, Any]) -> Optional[TestSample]:
        """딕셔너리에서 TestSample 생성"""
        if self.question_key not in data or self.answer_key not in data:
            return None

        return TestSample(
            id=sample_id,
            question=data[self.question_key],
            answer=data[self.answer_key],
            context=data.get('context'),
            category=data.get('category'),
            difficulty=data.get('difficulty'),
            language=data.get('language', self.language),
            metadata={k: v for k, v in data.items()
                     if k not in [self.question_key, self.answer_key, 'context', 'category', 'difficulty', 'language']}
        )

class KoreanMathLoader(JSONDatasetLoader):
    """한국어 수학 데이터셋 로더"""

    def __init__(self, dataset_path: str = None, **kwargs):
        if dataset_path is None:
            dataset_path = self._create_korean_math_dataset()

        super().__init__(dataset_path, question_key="problem", answer_key="solution", **kwargs)

    def _create_korean_math_dataset(self) -> str:
        """한국어 수학 데이터셋 생성"""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        dataset_path = data_dir / "korean_math.json"

        if not dataset_path.exists():
            korean_math_data = [
                {
                    "problem": "철수는 사탕 24개를 가지고 있습니다. 친구들에게 8개를 나누어 주었습니다. 철수에게 남은 사탕은 몇 개입니까?",
                    "solution": "24 - 8 = 16\n따라서 철수에게 남은 사탕은 16개입니다.",
                    "answer": "16",
                    "category": "기본_연산",
                    "difficulty": "easy",
                    "language": "ko"
                },
                {
                    "problem": "한 상자에 연필이 12자루씩 들어있습니다. 영희는 5상자를 샀습니다. 영희가 산 연필은 총 몇 자루입니까?",
                    "solution": "12 × 5 = 60\n따라서 영희가 산 연필은 총 60자루입니다.",
                    "answer": "60",
                    "category": "곱셈",
                    "difficulty": "easy",
                    "language": "ko"
                },
                {
                    "problem": "정사각형의 한 변의 길이가 7cm입니다. 이 정사각형의 둘레는 몇 cm입니까?",
                    "solution": "정사각형의 둘레 = 한 변의 길이 × 4\n7 × 4 = 28\n따라서 둘레는 28cm입니다.",
                    "answer": "28",
                    "category": "도형",
                    "difficulty": "medium",
                    "language": "ko"
                },
                {
                    "problem": "민수는 하루에 책을 15페이지씩 읽습니다. 7일 동안 읽은 페이지 수는 몇 페이지입니까?",
                    "solution": "하루에 15페이지씩 7일 동안 읽으면\n15 × 7 = 105\n따라서 105페이지입니다.",
                    "answer": "105",
                    "category": "곱셈",
                    "difficulty": "easy",
                    "language": "ko"
                },
                {
                    "problem": "가게에 사과가 48개 있었습니다. 오전에 19개를 팔고, 오후에 23개를 더 팔았습니다. 남은 사과는 몇 개입니까?",
                    "solution": "팔린 사과의 총 개수: 19 + 23 = 42개\n남은 사과: 48 - 42 = 6개\n따라서 남은 사과는 6개입니다.",
                    "answer": "6",
                    "category": "복합_연산",
                    "difficulty": "medium",
                    "language": "ko"
                }
            ]

            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(korean_math_data, f, ensure_ascii=False, indent=2)

        return str(dataset_path)

class KoreanQALoader(JSONDatasetLoader):
    """한국어 질의응답 데이터셋 로더"""

    def __init__(self, dataset_path: str = None, **kwargs):
        if dataset_path is None:
            dataset_path = self._create_korean_qa_dataset()

        super().__init__(dataset_path, question_key="question", answer_key="answer", **kwargs)

    def _create_korean_qa_dataset(self) -> str:
        """한국어 QA 데이터셋 생성"""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        dataset_path = data_dir / "korean_qa.json"

        if not dataset_path.exists():
            korean_qa_data = [
                {
                    "question": "대한민국의 수도는 어디입니까?",
                    "answer": "서울입니다.",
                    "category": "지리",
                    "difficulty": "easy",
                    "language": "ko"
                },
                {
                    "question": "한글을 창제한 조선의 왕은 누구입니까?",
                    "answer": "세종대왕입니다.",
                    "category": "역사",
                    "difficulty": "easy",
                    "language": "ko"
                },
                {
                    "question": "김치의 주재료는 무엇입니까?",
                    "answer": "배추입니다.",
                    "category": "음식",
                    "difficulty": "easy",
                    "language": "ko"
                },
                {
                    "question": "태양계에서 가장 큰 행성은 무엇입니까?",
                    "answer": "목성입니다.",
                    "category": "과학",
                    "difficulty": "medium",
                    "language": "ko"
                },
                {
                    "question": "물의 끓는점은 섭씨 몇 도입니까?",
                    "answer": "100도입니다.",
                    "category": "과학",
                    "difficulty": "easy",
                    "language": "ko"
                },
                {
                    "question": "조선왕조가 건국된 연도는 언제입니까?",
                    "answer": "1392년입니다.",
                    "category": "역사",
                    "difficulty": "medium",
                    "language": "ko"
                },
                {
                    "question": "한국의 전통 음악 중 판소리의 특징을 설명하세요.",
                    "answer": "판소리는 한 명의 소리꾼이 고수의 북 반주에 맞춰 긴 이야기를 창, 아니리, 발림으로 표현하는 전통 예술입니다.",
                    "category": "문화",
                    "difficulty": "hard",
                    "language": "ko"
                },
                {
                    "question": "인공지능과 머신러닝의 차이점은 무엇입니까?",
                    "answer": "인공지능은 인간의 지능을 모방하는 기술의 총칭이고, 머신러닝은 인공지능을 구현하는 방법 중 하나로 데이터를 통해 학습하는 기술입니다.",
                    "category": "기술",
                    "difficulty": "hard",
                    "language": "ko"
                }
            ]

            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(korean_qa_data, f, ensure_ascii=False, indent=2)

        return str(dataset_path)

class KoreanReasoningLoader(JSONDatasetLoader):
    """한국어 추론 문제 데이터셋 로더"""

    def __init__(self, dataset_path: str = None, **kwargs):
        if dataset_path is None:
            dataset_path = self._create_korean_reasoning_dataset()

        super().__init__(dataset_path, question_key="context", answer_key="answer", **kwargs)

    def _create_korean_reasoning_dataset(self) -> str:
        """한국어 추론 데이터셋 생성"""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        dataset_path = data_dir / "korean_reasoning.json"

        if not dataset_path.exists():
            reasoning_data = [
                {
                    "context": "철수가 집에서 학교까지 걸어가고 있다. 평소보다 30분 늦게 출발했는데",
                    "choices": [
                        "천천히 걸어간다",
                        "빨리 걸어간다",
                        "친구를 기다린다",
                        "집으로 돌아간다"
                    ],
                    "answer": "빨리 걸어간다",
                    "category": "상식추론",
                    "difficulty": "easy",
                    "language": "ko"
                },
                {
                    "context": "비가 오기 시작했는데 영희가 우산을 깜빡했다. 영희는",
                    "choices": [
                        "그냥 비를 맞으며 간다",
                        "건물 처마 밑에서 비를 피한다",
                        "뛰어서 빨리 간다",
                        "우산을 사러 간다"
                    ],
                    "answer": "뛰어서 빨리 간다",
                    "category": "상식추론",
                    "difficulty": "medium",
                    "language": "ko"
                },
                {
                    "context": "도서관에서 공부하고 있는데 옆 사람이 계속 큰 소리로 전화를 받고 있다. 가장 적절한 행동은",
                    "choices": [
                        "조용히 해달라고 정중하게 부탁한다",
                        "큰 소리로 항의한다",
                        "다른 자리로 이동한다",
                        "아무것도 하지 않는다"
                    ],
                    "answer": "조용히 해달라고 정중하게 부탁한다",
                    "category": "사회상식",
                    "difficulty": "medium",
                    "language": "ko"
                }
            ]

            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(reasoning_data, f, ensure_ascii=False, indent=2)

        return str(dataset_path)

class MultilingualMathLoader(JSONDatasetLoader):
    """다국어 수학 문제 로더"""

    def __init__(self, dataset_path: str = None, languages: List[str] = None, **kwargs):
        if dataset_path is None:
            dataset_path = self._create_multilingual_math_dataset()

        self.target_languages = languages or ["ko", "en", "ja", "zh"]
        super().__init__(dataset_path, **kwargs)

    def _create_multilingual_math_dataset(self) -> str:
        """다국어 수학 데이터셋 생성"""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        dataset_path = data_dir / "multilingual_math.json"

        if not dataset_path.exists():
            multilingual_data = [
                # 한국어
                {
                    "problem": "사과 5개와 배 3개가 있습니다. 과일은 총 몇 개입니까?",
                    "solution": "5 + 3 = 8개",
                    "answer": "8",
                    "language": "ko",
                    "category": "덧셈"
                },
                # 영어
                {
                    "problem": "There are 5 apples and 3 pears. How many fruits are there in total?",
                    "solution": "5 + 3 = 8 fruits",
                    "answer": "8",
                    "language": "en",
                    "category": "addition"
                },
                # 일본어
                {
                    "problem": "りんごが5個、なしが3個あります。果物は全部で何個ありますか？",
                    "solution": "5 + 3 = 8個",
                    "answer": "8",
                    "language": "ja",
                    "category": "足し算"
                },
                # 중국어
                {
                    "problem": "有5个苹果和3个梨。一共有多少个水果？",
                    "solution": "5 + 3 = 8个",
                    "answer": "8",
                    "language": "zh",
                    "category": "加法"
                }
            ]

            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(multilingual_data, f, ensure_ascii=False, indent=2)

        return str(dataset_path)

    def load_samples(self) -> List[TestSample]:
        """다국어 샘플 로드"""
        all_samples = super().load_samples()

        # 지정된 언어만 필터링
        filtered_samples = [
            sample for sample in all_samples
            if sample.language in self.target_languages
        ]

        return filtered_samples

class LlamaSpecificLoader(JSONDatasetLoader):
    """Llama 모델 특화 프롬프트 로더"""

    def __init__(self, dataset_path: str, model_family: str = "llama", **kwargs):
        super().__init__(dataset_path, **kwargs)
        self.model_family = model_family

    def _create_sample_from_dict(self, sample_id: str, data: Dict[str, Any]) -> Optional[TestSample]:
        """Llama 스타일 프롬프트로 변환"""
        sample = super()._create_sample_from_dict(sample_id, data)
        if not sample:
            return None

        # Llama 프롬프트 템플릿 적용
        if self.model_family.lower() == "llama":
            formatted_question = self._format_llama_prompt(sample.question, sample.context)
            sample.question = formatted_question

        return sample

    def _format_llama_prompt(self, question: str, context: str = None) -> str:
        """Llama 모델용 프롬프트 포맷팅"""
        if context:
            return f"<s>[INST] <<SYS>>\n주어진 맥락을 바탕으로 정확하고 도움이 되는 답변을 제공하세요.\n<</SYS>>\n\n맥락: {context}\n\n질문: {question} [/INST]"
        else:
            return f"<s>[INST] {question} [/INST]"

class MistralSpecificLoader(JSONDatasetLoader):
    """Mistral 모델 특화 프롬프트 로더"""

    def _create_sample_from_dict(self, sample_id: str, data: Dict[str, Any]) -> Optional[TestSample]:
        """Mistral 스타일 프롬프트로 변환"""
        sample = super()._create_sample_from_dict(sample_id, data)
        if not sample:
            return None

        # Mistral 프롬프트 템플릿 적용
        formatted_question = self._format_mistral_prompt(sample.question, sample.context)
        sample.question = formatted_question

        return sample

    def _format_mistral_prompt(self, question: str, context: str = None) -> str:
        """Mistral 모델용 프롬프트 포맷팅"""
        if context:
            return f"<s>[INST] 맥락: {context}\n\n질문: {question} [/INST]"
        else:
            return f"<s>[INST] {question} [/INST]"

class GemmaSpecificLoader(JSONDatasetLoader):
    """Gemma 모델 특화 프롬프트 로더"""

    def _create_sample_from_dict(self, sample_id: str, data: Dict[str, Any]) -> Optional[TestSample]:
        """Gemma 스타일 프롬프트로 변환"""
        sample = super()._create_sample_from_dict(sample_id, data)
        if not sample:
            return None

        # Gemma 프롬프트 템플릿 적용
        formatted_question = self._format_gemma_prompt(sample.question, sample.context)
        sample.question = formatted_question

        return sample

    def _format_gemma_prompt(self, question: str, context: str = None) -> str:
        """Gemma 모델용 프롬프트 포맷팅"""
        if context:
            return f"<start_of_turn>user\n맥락: {context}\n\n질문: {question}<end_of_turn>\n<start_of_turn>model\n"
        else:
            return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"

class QwenSpecificLoader(JSONDatasetLoader):
    """Qwen 모델 특화 프롬프트 로더"""

    def _create_sample_from_dict(self, sample_id: str, data: Dict[str, Any]) -> Optional[TestSample]:
        """Qwen 스타일 프롬프트로 변환"""
        sample = super()._create_sample_from_dict(sample_id, data)
        if not sample:
            return None

        # Qwen 프롬프트 템플릿 적용
        formatted_question = self._format_qwen_prompt(sample.question, sample.context)
        sample.question = formatted_question

        return sample

    def _format_qwen_prompt(self, question: str, context: str = None) -> str:
        """Qwen 모델용 프롬프트 포맷팅"""
        if context:
            return f"<|im_start|>system\n당신은 도움이 되는 AI 어시스턴트입니다.<|im_end|>\n<|im_start|>user\n맥락: {context}\n\n질문: {question}<|im_end|>\n<|im_start|>assistant\n"
        else:
            return f"<|im_start|>system\n당신은 도움이 되는 AI 어시스턴트입니다.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

class DatasetManager:
    """데이터셋 관리 클래스"""

    def __init__(self):
        self.loaders = {}
        self.register_default_loaders()

    def register_default_loaders(self):
        """기본 로더들 등록"""
        self.loaders.update({
            'json': JSONDatasetLoader,
            'korean_math': KoreanMathLoader,
            'korean_qa': KoreanQALoader,
            'korean_reasoning': KoreanReasoningLoader,
            'multilingual_math': MultilingualMathLoader,
            'llama_specific': LlamaSpecificLoader,
            'mistral_specific': MistralSpecificLoader,
            'gemma_specific': GemmaSpecificLoader,
            'qwen_specific': QwenSpecificLoader
        })

    def register_loader(self, name: str, loader_class: type):
        """새로운 로더 등록"""
        self.loaders[name] = loader_class

    def create_loader(self, loader_type: str, **kwargs) -> BaseDatasetLoader:
        """로더 생성"""
        if loader_type not in self.loaders:
            raise ValueError(f"Unknown loader type: {loader_type}")

        return self.loaders[loader_type](**kwargs)

    def load_dataset(self, loader_type: str, **kwargs) -> List[TestSample]:
        """데이터셋 로드"""
        loader = self.create_loader(loader_type, **kwargs)
        return loader.load_samples()

    def get_model_specific_loader(self, model_family: str, dataset_path: str, **kwargs) -> BaseDatasetLoader:
        """모델별 특화 로더 반환"""
        model_family_lower = model_family.lower()

        if 'llama' in model_family_lower:
            return LlamaSpecificLoader(dataset_path, model_family="llama", **kwargs)
        elif 'mistral' in model_family_lower:
            return MistralSpecificLoader(dataset_path, **kwargs)
        elif 'gemma' in model_family_lower:
            return GemmaSpecificLoader(dataset_path, **kwargs)
        elif 'qwen' in model_family_lower:
            return QwenSpecificLoader(dataset_path, **kwargs)
        else:
            return JSONDatasetLoader(dataset_path, **kwargs)

class DatasetAnalyzer:
    """데이터셋 분석 클래스"""

    @staticmethod
    def analyze_dataset(samples: List[TestSample]) -> Dict[str, Any]:
        """데이터셋 통계 분석"""
        if not samples:
            return {}

        stats = {
            'total_samples': len(samples),
            'languages': {},
            'categories': {},
            'difficulties': {},
            'question_lengths': [],
            'answer_lengths': []
        }

        for sample in samples:
            # 언어 통계
            if sample.language:
                stats['languages'][sample.language] = stats['languages'].get(sample.language, 0) + 1

            # 카테고리 통계
            if sample.category:
                stats['categories'][sample.category] = stats['categories'].get(sample.category, 0) + 1

            # 난이도 통계
            if sample.difficulty:
                stats['difficulties'][sample.difficulty] = stats['difficulties'].get(sample.difficulty, 0) + 1

            # 길이 통계
            stats['question_lengths'].append(len(sample.question))
            stats['answer_lengths'].append(len(sample.answer))

        # 길이 통계 요약
        if stats['question_lengths']:
            stats['avg_question_length'] = sum(stats['question_lengths']) / len(stats['question_lengths'])
            stats['max_question_length'] = max(stats['question_lengths'])
            stats['min_question_length'] = min(stats['question_lengths'])

        if stats['answer_lengths']:
            stats['avg_answer_length'] = sum(stats['answer_lengths']) / len(stats['answer_lengths'])
            stats['max_answer_length'] = max(stats['answer_lengths'])
            stats['min_answer_length'] = min(stats['answer_lengths'])

        return stats

    @staticmethod
    def print_dataset_info(samples: List[TestSample]):
        """데이터셋 정보 출력"""
        stats = DatasetAnalyzer.analyze_dataset(samples)

        print(f"=== 데이터셋 분석 ===")
        print(f"총 샘플 수: {stats.get('total_samples', 0)}")
        print(f"평균 질문 길이: {stats.get('avg_question_length', 0):.1f}자")
        print(f"평균 답변 길이: {stats.get('avg_answer_length', 0):.1f}자")

        if stats.get('languages'):
            print(f"\n언어별 분포:")
            for language, count in stats['languages'].items():
                print(f"  {language}: {count}개")

        if stats.get('categories'):
            print(f"\n카테고리별 분포:")
            for category, count in stats['categories'].items():
                print(f"  {category}: {count}개")

        if stats.get('difficulties'):
            print(f"\n난이도별 분포:")
            for difficulty, count in stats['difficulties'].items():
                print(f"  {difficulty}: {count}개")

# 사용 예시
if __name__ == "__main__":
    # 데이터셋 매니저 생성
    dataset_manager = DatasetManager()

    # 한국어 수학 데이터셋 로드
    print("=== 한국어 수학 데이터셋 ===")
    korean_math_samples = dataset_manager.load_dataset('korean_math', num_samples=3)
    DatasetAnalyzer.print_dataset_info(korean_math_samples)

    print(f"\n첫 번째 샘플:")
    print(f"문제: {korean_math_samples[0].question}")
    print(f"답: {korean_math_samples[0].answer}")
    print(f"카테고리: {korean_math_samples[0].category}")

    # 한국어 QA 데이터셋 로드
    print(f"\n=== 한국어 QA 데이터셋 ===")
    korean_qa_samples = dataset_manager.load_dataset('korean_qa', num_samples=3)
    DatasetAnalyzer.print_dataset_info(korean_qa_samples)

    # 다국어 수학 데이터셋 로드
    print(f"\n=== 다국어 수학 데이터셋 ===")
    multilingual_samples = dataset_manager.load_dataset('multilingual_math')
    DatasetAnalyzer.print_dataset_info(multilingual_samples)

    # 모델별 특화 로더 테스트
    print(f"\n=== 모델별 특화 로더 테스트 ===")

    # Llama 특화 로더
    llama_loader = dataset_manager.get_model_specific_loader(
        'llama2-7b',
        'data/korean_qa.json'
    )
    llama_samples = llama_loader.load_samples()
    if llama_samples:
        print(f"Llama 프롬프트 예시:")
        print(f"{llama_samples[0].question[:100]}...")

    # Gemma 특화 로더
    gemma_loader = dataset_manager.get_model_specific_loader(
        'gemma-7b',
        'data/korean_qa.json'
    )
    gemma_samples = gemma_loader.load_samples()
    if gemma_samples:
        print(f"\nGemma 프롬프트 예시:")
        print(f"{gemma_samples[0].question[:100]}...")