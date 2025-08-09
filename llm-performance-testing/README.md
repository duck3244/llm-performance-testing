# 🚀 오픈소스 LLM 추론 성능 최적화 시스템

Llama, Mistral, Gemma, Qwen 등 오픈소스 LLM의 추론 성능을 체계적으로 최적화하고 벤치마크하는 종합 도구입니다.

## ✨ 주요 기능

- **🔧 파라미터 최적화**: 베이지안/그리드서치/진화 알고리즘을 통한 자동 최적화
- **⚡ 성능 벤치마크**: 속도, 메모리, 비용 효율성 종합 분석
- **💻 하드웨어 최적화**: GPU 메모리에 따른 자동 설정 추천
- **📊 실시간 모니터링**: CPU/GPU/메모리 사용량 실시간 추적
- **🌐 다양한 추론 엔진**: Transformers, vLLM, Ollama, TGI 지원
- **🇰🇷 한국어 특화**: 한국어 수학, QA, 추론 문제 전용 평가자
- **📈 인터랙티브 시각화**: Plotly 기반 대시보드 및 상세 분석 차트
- **🤖 모델별 프롬프트**: Llama, Mistral, Gemma, Qwen 최적화된 프롬프트 템플릿

## 🎯 지원 모델

### 🦙 Llama 계열
- Meta Llama 2 (7B, 13B, 70B)
- Code Llama
- Llama 2-Chat

### 🌟 Mistral 계열  
- Mistral 7B
- Mixtral 8x7B
- Mistral-Instruct

### 💎 Gemma 계열
- Gemma 2B, 7B
- Gemma-IT (Instruction Tuned)

### 🔥 Qwen 계열
- Qwen 1.5 (0.5B~72B)
- Qwen-Chat
- Qwen-Math

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Interface │────│ Performance      │────│ Model Interfaces│
│   (main.py)     │    │ Optimizer        │    │ (Transformers,  │
└─────────────────┘    │ (test_runner.py) │    │  vLLM, Ollama)  │
                       └──────────────────┘    └─────────────────┘
                                │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dataset       │────│ Config Manager   │────│ Hardware        │
│   Loaders       │    │ (config.py)      │    │ Monitor         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Evaluators    │────│ Result           │────│ Visualization   │
│   (Korean NLP)  │    │ Storage          │    │ Dashboard       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone <repository-url>
cd llm-optimization-system

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# GPU 가속을 위한 PyTorch 설치 (CUDA 11.8 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 시스템 초기화

```bash
# 하드웨어 자동 감지와 함께 초기화
python main.py init --auto-detect

# 하드웨어 정보 확인
python main.py hardware --model-size 7b
```

### 3. 모델 설정

`llm_config.json` 파일에서 모델 경로를 설정:

```json
{
  "models": {
    "llama2-7b": {
      "name": "llama2-7b",
      "model_path": "meta-llama/Llama-2-7b-chat-hf",
      "model_type": "transformers",
      "device": "auto",
      "dtype": "float16",
      "load_in_4bit": false
    }
  }
}
```

### 4. 첫 번째 최적화 실행

```bash
# 파라미터 최적화 (베이지안 방법)
python main.py optimize --model llama2-7b --dataset korean_math --strategy bayesian

# 결과 확인
python main.py visualize --type optimization
```

## 📖 상세 사용법

### 🔧 파라미터 최적화

```bash
# 베이지안 최적화 (권장)
python main.py optimize --model mistral-7b --dataset korean_qa \
  --strategy bayesian --trials 50 --samples 100

# 그리드 서치 (빠른 테스트용)
python main.py optimize --model gemma-7b --dataset korean_reasoning \
  --strategy grid_search --samples 50

# 진화 알고리즘 (실험적)
python main.py optimize --model qwen-7b --dataset multilingual_math \
  --strategy evolutionary --trials 30
```

### ⚡ 성능 벤치마크

```bash
# 기본 벤치마크
python main.py benchmark --model llama2-7b --dataset korean_math \
  --samples 200 --iterations 3

# 커스텀 파라미터로 벤치마크
python main.py benchmark --model mistral-7b --dataset korean_qa \
  --temperature 0.1 --top-p 0.9 --max-tokens 512
```

### ⚖️ 모델 비교

```bash
# 정확도 기준 비교
python main.py compare --models llama2-7b mistral-7b gemma-7b \
  --dataset korean_math --metric accuracy

# 속도 기준 비교
python main.py compare --models llama2-7b mistral-7b \
  --dataset korean_qa --metric speed

# 비용 효율성 비교
python main.py compare --models gemma-7b qwen-7b \
  --dataset korean_reasoning --metric cost
```

### 📊 결과 시각화

```bash
# 최적화 결과 분석
python main.py visualize --type optimization --filter-model llama2-7b

# 벤치마크 결과 분석
python main.py visualize --type benchmark

# 종합 대시보드
python main.py visualize --type dashboard

# 모델 비교 차트
python main.py visualize --type comparison
```

### 🔍 결과 분석

```bash
# 전체 결과 분석
python main.py analyze --report

# 특정 모델 분석
python main.py analyze --model llama2-7b --report

# 특정 데이터셋 분석  
python main.py analyze --dataset korean_math --report
```

## 💡 최적화 전략 가이드

### 🎯 정확도 우선 설정
```
Temperature: 0.0 - 0.3
Top-p: 0.1 - 0.5
Top-k: 1 - 20
용도: 수학 문제, 사실 확인, 번역
```

### 🎨 창의성 우선 설정
```
Temperature: 0.7 - 1.0
Top-p: 0.8 - 0.95
Top-k: 40 - 50
용도: 창작, 브레인스토밍, 대화
```

### ⚡ 속도 우선 설정
```
- vLLM 엔진 사용
- 동적 배칭 활성화
- Flash Attention 사용
- 4-bit 양자화 적용
```

### 💾 메모리 효율성 설정
```
- load_in_4bit: true
- gradient_checkpointing: true
- cpu_offload: true (필요시)
```

## 🔧 하드웨어별 추천 설정

### 🖥️ 단일 GPU (8-16GB)
```json
{
  "model_size": "7B",
  "dtype": "float16",
  "load_in_4bit": true,
  "max_batch_size": 4
}
```

### 🖥️ 고성능 GPU (24GB+)
```json
{
  "model_size": "13B",
  "dtype": "float16", 
  "load_in_4bit": false,
  "max_batch_size": 16
}
```

### 🖥️ 다중 GPU (80GB+)
```json
{
  "model_size": "70B",
  "tensor_parallel_size": 4,
  "pipeline_parallel_size": 1,
  "dtype": "bfloat16"
}
```

### 💻 CPU 전용
```json
{
  "model_size": "7B",
  "device": "cpu",
  "dtype": "float32",
  "load_in_8bit": true
}
```

## 📊 데이터셋 정보

### 🇰🇷 한국어 데이터셋
- **korean_math**: 한국어 수학 문제 (100개)
- **korean_qa**: 한국어 질의응답 (150개)  
- **korean_reasoning**: 한국어 상식 추론 (80개)

### 🌍 다국어 데이터셋
- **multilingual_math**: 4개 언어 수학 문제
- **gsm8k**: 영어 수학 문제 (표준 벤치마크)
- **hellaswag**: 영어 상식 추론

### 📁 커스텀 데이터셋 추가

```python
# data/my_dataset.json 형식
[
  {
    "question": "질문 내용",
    "answer": "정답",
    "category": "카테고리",
    "difficulty": "easy|medium|hard",
    "language": "ko"
  }
]
```

```bash
# 설정 파일에 데이터셋 추가
# llm_config.json의 tests 섹션에 추가
```

## 🎨 시각화 예시

### 📈 최적화 트렌드
- 모델별 성능 변화 추이
- 파라미터 상관관계 분석
- 하드웨어 사용량 히트맵

### ⚡ 성능 벤치마크
- 처리 속도 비교 막대 차트
- 지연시간 분포 박스 플롯
- 메모리 vs 성능 산점도

### 🏆 모델 비교 레이더
- 정확도, 속도, 효율성, 비용 종합 비교
- 모델별 강점/약점 시각화

## 🛠️ 고급 설정

### 🔧 추론 엔진별 설정

#### vLLM (고성능)
```json
{
  "model_type": "vllm",
  "tensor_parallel_size": 2,
  "gpu_memory_utilization": 0.9,
  "max_model_len": 4096
}
```

#### Ollama (간편)
```json
{
  "model_type": "ollama", 
  "base_url": "http://localhost:11434",
  "model_path": "llama2:7b"
}
```

#### TGI (프로덕션)
```json
{
  "model_type": "tgi",
  "base_url": "http://localhost:3000",
  "max_batch_prefill_tokens": 4096
}
```

### 🎛️ 고급 최적화 옵션

```json
{
  "optimization": {
    "enable_torch_compile": true,
    "enable_bettertransformer": true,
    "dynamic_batching": true,
    "gradient_checkpointing": true
  }
}
```

## 📈 성능 모니터링

### 실시간 메트릭
- CPU/GPU 사용률
- 메모리 사용량
- 토큰 처리 속도
- 요청 처리량

### 자동 리포트
- 일일/주간 성능 요약
- 이상 감지 알림
- 최적화 추천사항

## 🐛 문제 해결

### 일반적인 오류

#### CUDA 메모리 부족
```bash
# 해결책: 배치 크기 줄이기, 양자화 사용
python main.py hardware  # 현재 GPU 메모리 확인
```

#### 모델 로딩 실패
```bash
# 해결책: 모델 경로 확인, 권한 설정
python main.py list --type models  # 모델 상태 확인
```

#### 성능 저하
```bash
# 해결책: 하드웨어 점검, 파라미터 재최적화
python main.py analyze --model [모델명] --report
```

### 로그 확인
```bash
# 상세 로그 출력
export PYTHONPATH=. && python -u main.py optimize --model llama2-7b --dataset korean_math -v

# 결과 파일 위치
ls optimization_results/
ls visualizations/
```

```bash
# 개발 환경 설정
pip install -r requirements.txt
pip install pytest black flake8 mypy

# 테스트 실행
pytest tests/

# 코드 포맷팅
black . && flake8 .
```

## 📄 라이선스

MIT License - 자세한 내용은 LICENSE 파일 참조

---

💡 **팁**: 처음 사용하시는 경우 `python main.py`를 실행하여 시스템 상태를 확인하고 빠른 시작 가이드를 참조하세요!

🚀 **성능 최적화로 더 나은 LLM 경험을 만들어보세요!**