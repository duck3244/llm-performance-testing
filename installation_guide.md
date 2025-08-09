# 🛡️ 안전한 오픈소스 LLM 최적화 시스템 설치 가이드

## 📋 시스템 요구사항

### 최소 요구사항
- **Python**: 3.8 이상 (3.9-3.11 권장)
- **메모리**: 8GB RAM 이상
- **저장공간**: 20GB 이상 여유공간
- **운영체제**: Windows 10+, Ubuntu 18.04+, macOS 10.15+

### 권장 사양
- **Python**: 3.10
- **메모리**: 16GB RAM 이상
- **GPU**: CUDA 11.8+ 지원 GPU (8GB+ VRAM)
- **저장공간**: 50GB 이상 SSD

---

## 🚀 빠른 설치 (권장)

### 1단계: 저장소 클론
```bash
git clone <repository-url>
cd safe-llm-optimizer
```

### 2단계: 가상환경 생성
```bash
# Python venv 사용
python -m venv venv

# 활성화
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3단계: 안전한 의존성 설치
```bash
# 기본 패키지 설치
pip install -r requirements.txt

# GPU 가속 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU 가속 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU 전용
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4단계: 시스템 초기화
```bash
python safe_main.py init --auto-detect
```

### 5단계: 설치 확인
```bash
python safe_main.py status
```

---

## 🔧 상세 설치 가이드

### A. 환경별 설치

#### 🐧 Linux (Ubuntu/Debian)
```bash
# 시스템 패키지 업데이트
sudo apt update && sudo apt upgrade -y

# Python 및 개발 도구 설치
sudo apt install python3.10 python3.10-venv python3-pip git -y

# CUDA 지원 (선택적)
# NVIDIA 드라이버 설치
sudo apt install nvidia-driver-535 -y

# 저장소 클론 및 설정
git clone <repository-url>
cd safe-llm-optimizer
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 🪟 Windows
```cmd
# Git for Windows 설치 필요
# Python 3.10 설치 (python.org에서 다운로드)

# PowerShell에서 실행
git clone <repository-url>
cd safe-llm-optimizer
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 🍎 macOS
```bash
# Homebrew 설치 (없는 경우)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 설치
brew install python@3.10 git

# 저장소 설정
git clone <repository-url>
cd safe-llm-optimizer
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Apple Silicon Mac의 경우
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### B. GPU 설정

#### NVIDIA CUDA
```bash
# CUDA 버전 확인
nvidia-smi

# CUDA 11.8 (권장)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install bitsandbytes  # 양자화 지원

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes

# 설치 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Apple Silicon (M1/M2)
```bash
# MPS 지원 PyTorch
pip install torch torchvision torchaudio

# 설치 확인
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### C. 선택적 패키지 설치

#### 고성능 추론 엔진
```bash
# vLLM (CUDA 환경에서만)
pip install vllm>=0.2.7

# Ollama Python 클라이언트
pip install ollama-python

# Text Generation Inference
pip install text-generation>=0.6.0
```

#### 고급 최적화 도구
```bash
# 베이지안 최적화 (권장)
pip install optuna>=3.4.0

# 시각화 도구
pip install plotly>=5.17.0 kaleido>=0.2.1

# 웹 인터페이스
pip install streamlit>=1.28.0
```

---

## ⚙️ 설정 및 검증

### 1. 하드웨어 감지
```bash
python safe_main.py hardware
```

**예상 출력:**
```
💻 하드웨어 정보 분석

🔍 감지된 하드웨어:
   플랫폼: linux
   CUDA 사용 가능: ✅
   GPU 개수: 1
   총 메모리: 32GB
   GPU 0: NVIDIA GeForce RTX 4090 (24GB, CC 8.9)

🎯 안전한 모델 추천:
   ✅ 70B 모델까지 실행 가능 (4-bit 양자화 권장)
```

### 2. 모델 목록 확인
```bash
python safe_main.py list --type models
```

### 3. 첫 번째 테스트
```bash
# 안전 모드로 간단한 최적화
python safe_main.py optimize --model qwen2.5-7b --dataset korean_math --samples 5 --safe

# 성공 시 출력:
# ✅ 최적화 완료!
#    최고 점수: 0.850
#    소요 시간: 45.2초
```

---

## 🔧 문제 해결

### 일반적인 설치 오류

#### 1. `ModuleNotFoundError: No module named 'torch'`
**해결:**
```bash
pip install torch torchvision torchaudio
```

#### 2. `CUDA out of memory`
**해결:**
```bash
# 더 작은 모델 사용
python safe_main.py optimize --model qwen2.5-7b --dataset korean_math --samples 5 --safe

# 또는 CPU 모드
export CUDA_VISIBLE_DEVICES=""
```

#### 3. `ImportError: No module named 'optuna'`
**해결:**
```bash
pip install optuna
# 또는 그리드 서치 사용
python safe_main.py optimize --strategy grid_search
```

#### 4. `Permission denied` (Linux/Mac)
**해결:**
```bash
# 가상환경 권한 확인
chmod +x venv/bin/activate
source venv/bin/activate

# 또는 sudo 없이 설치
pip install --user -r requirements.txt
```

#### 5. `SSL Certificate` 오류
**해결:**
```bash
# pip 업그레이드
pip install --upgrade pip

# 신뢰할 수 있는 호스트 추가
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### 플랫폼별 문제

#### Windows 관련
```cmd
# Long Path 활성화 (관리자 권한)
# 레지스트리: HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
# LongPathsEnabled = 1

# PowerShell 실행 정책
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### macOS 관련
```bash
# Xcode Command Line Tools
xcode-select --install

# OpenMP 문제 해결
brew install libomp

# M1/M2 Mac 환경변수
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
```

#### Linux 관련
```bash
# GLIBC 버전 문제
ldd --version

# CUDA 경로 설정
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## 🎯 성능 최적화 설정

### 메모리 최적화
```bash
# 환경변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=./cache
```

### GPU 최적화
```bash
# GPU 선택
export CUDA_VISIBLE_DEVICES=0

# 메모리 단편화 방지
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,roundup_power2_divisions:16
```

### CPU 최적화
```bash
# OpenMP 스레드 수 (CPU 코어 수의 절반)
export OMP_NUM_THREADS=8

# Intel 최적화 (Intel CPU)
pip install intel-extension-for-pytorch
```

---

## 📊 설치 검증 체크리스트

### ✅ 기본 확인사항
- [ ] Python 3.8+ 설치됨
- [ ] 가상환경 활성화됨
- [ ] 모든 필수 패키지 설치됨
- [ ] `python safe_main.py status` 정상 실행
- [ ] 하드웨어 정보 정상 감지

### ✅ GPU 확인사항 (해당 시)
- [ ] NVIDIA 드라이버 설치됨
- [ ] CUDA toolkit 설치됨
- [ ] `torch.cuda.is_available()` = True
- [ ] GPU 메모리 정보 표시됨

### ✅ 기능 확인사항
- [ ] 모델 목록 표시됨
- [ ] 데이터셋 로드 가능
- [ ] 간단한 최적화 실행 가능
- [ ] 결과 파일 저장됨

---

## 🆘 지원 및 문의

### 로그 확인
```bash
# 상세 로그로 실행
python safe_main.py optimize --model qwen2.5-7b --dataset korean_math --debug

# 로그 파일 위치
cat logs/llm_optimizer_20231201.log
```

### 시스템 정보 수집
```bash
# 시스템 정보 출력
python safe_main.py hardware > system_info.txt
python safe_main.py status >> system_info.txt

# Python 환경 정보
pip list > installed_packages.txt
```

### 문제 보고 시 포함할 정보
1. 운영체제 및 버전
2. Python 버전
3. GPU 정보 (있는 경우)
4. 오류 메시지 전체
5. 실행한 명령어
6. `system_info.txt` 내용

---

## 🎉 설치 완료!

설치가 완료되면 다음 단계로 진행하세요:

1. **첫 번째 최적화**: [사용자 가이드](USER_GUIDE.md) 참조
2. **고급 설정**: [설정 가이드](CONFIG_GUIDE.md) 참조
3. **API 문서**: [API 문서](API_DOCS.md) 참조

**성공적인 설치를 축하합니다! 🎊**