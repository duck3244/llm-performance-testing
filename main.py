class SafeCLI:
    """안전한 CLI 인터페이스"""

    def __init__(self):
        self.logger = None
        self.config_manager = None
        self.optimizer = None
        self._initialized = False

    def _ensure_imports(self):
        """필요한 모듈들이 import되었는지 확인"""
        if not IMPORTS_OK:
            print(f"❌ {IMPORT_ERROR}")
            print("\n📦 누락된 의존성을 설치하세요:")
            print("   pip install torch transformers numpy pandas optuna")
            print("\n📁 필요한 파일들을 확인하세요:")
            print("   - safe_config.py")
            print("   - safe_model_interface.py")
            print("   - safe_optimizer.py")
            print("   - dataset_loader.py")
            sys.exit(1)

    def _initialize(self):
        """CLI 초기화"""
        if self._initialized:
            return

        self._ensure_imports()

        # 이제 안전하게 import 가능
        global ConfigManager, HardwareDetector, SafeOptimizer
        from config import ConfigManager, HardwareDetector
        from optimizer import SafeOptimizer

        self.logger = logging.getLogger(__name__)
        self._initialized = True

    def create_argument_parser(self) -> argparse.ArgumentParser:
        """CLI 인자 파서 생성"""
        parser = argparse.ArgumentParser(
            description='안전한 오픈소스 LLM 추론 성능 최적화 시스템',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
안전한 사용 예시:
  python safe_main.py init --auto-detect              # 시스템 초기화
  python safe_main.py hardware                        # 하드웨어 정보 확인
  python safe_main.py optimize --model qwen2.5-7b --dataset korean_math --samples 20 --safe
  python safe_main.py benchmark --model qwen2.5-7b --dataset korean_qa --samples 30
  python safe_main.py compare --models qwen2.5-7b llama2-7b --dataset korean_math
  python safe_main.py status                          # 시스템 상태 확인

안전 모드 옵션:
  --safe: 메모리와 성능을 제한하여 안전하게 실행
  --debug: 상세한 디버그 정보 출력
  --timeout: 작업 시간 제한 (초)
            """
        )

        # 전역 옵션
        parser.add_argument('--safe', action='store_true', help='안전 모드로 실행')
        parser.add_argument('--debug', action='store_true', help='디버그 모드')
        parser.add_argument('--timeout', type=int, default=3600, help='작업 시간 제한 (초)')

        subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')

        # init 명령어
        init_parser = subparsers.add_parser('init', help='시스템 초기화')
        init_parser.add_argument('--force', action='store_true', help='기존 설정 덮어쓰기')
        init_parser.add_argument('--auto-detect', action='store_true', help='하드웨어 자동 감지')

        # hardware 명령어
        hardware_parser = subparsers.add_parser('hardware', help='하드웨어 정보 확인')
        hardware_parser.add_argument('--model-size', choices=['1b', '3b', '7b', '13b', '30b', '70b'],
                                     help='모델 크기별 추천')

        # optimize 명령어
        optimize_parser = subparsers.add_parser('optimize', help='파라미터 최적화')
        optimize_parser.add_argument('--model', required=True, help='모델 이름')
        optimize_parser.add_argument('--dataset', required=True, help='데이터셋 이름')
        optimize_parser.add_argument('--strategy', choices=['optuna', 'grid_search'],
                                     default='grid_search', help='최적화 전략')
        optimize_parser.add_argument('--trials', type=int, default=10, help='최적화 시도 횟수')
        optimize_parser.add_argument('--samples', type=int, default=20, help='테스트 샘플 수')
        optimize_parser.add_argument('--evaluator', choices=['exact_match', 'contains', 'similarity'],
                                     default='similarity', help='평가자 유형')

        # benchmark 명령어
        benchmark_parser = subparsers.add_parser('benchmark', help='성능 벤치마크')
        benchmark_parser.add_argument('--model', required=True, help='모델 이름')
        benchmark_parser.add_argument('--dataset', required=True, help='데이터셋 이름')
        benchmark_parser.add_argument('--samples', type=int, default=30, help='테스트 샘플 수')
        benchmark_parser.add_argument('--iterations', type=int, default=1, help='반복 횟수')
        benchmark_parser.add_argument('--temperature', type=float, default=0.1, help='Temperature')
        benchmark_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p')
        benchmark_parser.add_argument('--max-tokens', type=int, default=200, help='최대 토큰 수')

        # compare 명령어
        compare_parser = subparsers.add_parser('compare', help='모델 비교')
        compare_parser.add_argument('--models', nargs='+', required=True, help='비교할 모델들')
        compare_parser.add_argument('--dataset', required=True, help='데이터셋 이름')
        compare_parser.add_argument('--samples', type=int, default=20, help='테스트 샘플 수')
        compare_parser.add_argument('--metric', choices=['accuracy', 'speed', 'efficiency'],
                                    default='accuracy', help='비교 기준')

        # list 명령어
        list_parser = subparsers.add_parser('list', help='정보 조회')
        list_parser.add_argument('--type', choices=['models', 'datasets', 'results'],
                                 default='models', help='조회할 정보 유형')

        # status 명령어
        status_parser = subparsers.add_parser('status', help='시스템 상태 확인')

        # clean 명령어
        clean_parser = subparsers.add_parser('clean', help='시스템 정리')
        clean_parser.add_argument('--cache', action='store_true', help='캐시 정리')
        clean_parser.add_argument('--logs', action='store_true', help='로그 정리')
        clean_parser.add_argument('--results', action='store_true', help='결과 정리')
        clean_parser.add_argument('--all', action='store_true', help='모든 정리')

        return parser

    def print_banner(self):
        """시스템 배너 출력"""
        banner = """
╭─────────────────────────────────────────────────────────────╮
│          🛡️ 안전한 오픈소스 LLM 추론 성능 최적화 시스템         │
│              Safe Open Source LLM Optimizer               │
│                    ✅ Security Enhanced                    │
╰─────────────────────────────────────────────────────────────╯
"""
        print(banner)

    def check_system_requirements(self) -> bool:
        """시스템 요구사항 확인"""
        requirements_met = True
        issues = []

        # Python 버전 확인
        if sys.version_info < (3, 8):
            issues.append(f"Python 3.8+ 필요 (현재: {sys.version})")
            requirements_met = False

        # 필수 패키지 확인
        required_packages = {
            'torch': 'PyTorch',
            'transformers': 'HuggingFace Transformers',
            'numpy': 'NumPy',
            'psutil': 'psutil'
        }

        missing_packages = []
        for package, description in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(f"{package} ({description})")

        if missing_packages:
            issues.append(f"누락된 필수 패키지: {', '.join(missing_packages)}")
            requirements_met = False

        # 선택적 패키지 확인
        optional_packages = {
            'optuna': 'Optuna (고급 최적화용)',
            'plotly': 'Plotly (시각화용)'
        }

        missing_optional = []
        for package, description in optional_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(f"{package} ({description})")

        if missing_optional:
            print(f"⚠️ 선택적 패키지 누락: {', '.join(missing_optional)}")
            print("  일부 기능이 제한될 수 있습니다.")

        # 하드웨어 정보
        try:
            if IMPORTS_OK:
                hardware_info = HardwareDetector.detect_hardware()
                if hardware_info['cuda_available']:
                    gpu_count = hardware_info['cuda_device_count']
                    total_memory = sum(
                        hardware_info.get(f'gpu_{i}_memory', 0)
                        for i in range(gpu_count)
                    )
                    print(f"✅ GPU 감지: {gpu_count}개, 총 메모리: {total_memory}GB")
                else:
                    print("⚠️ CUDA 사용 불가 - CPU 모드로 실행됩니다.")
        except Exception as e:
            print(f"⚠️ 하드웨어 정보 확인 불가: {e}")

        # 오류 출력
        if issues:
            print(f"❌ 시스템 요구사항 미충족:")
            for issue in issues:
                print(f"   - {issue}")

            print(f"\n💡 해결 방법:")
            if missing_packages:
                packages = [p.split(' ')[0] for p in missing_packages]
                print(f"   pip install {' '.join(packages)}")

        return requirements_met

    def show_system_status(self):
        """시스템 상태 표시"""
        print("🔧 시스템 상태:")

        # 설정 파일 확인
        config_file = Path("llm_config.json")
        if config_file.exists():
            try:
                if not self.config_manager:
                    self.config_manager = ConfigManager()
                model_count = len(self.config_manager.model_configs)
                print(f"   ✅ 설정 파일: {model_count}개 모델 등록됨")

                # 설정 검증
                validation_results = self.config_manager.validate_all_configs()
                if validation_results:
                    print(f"   ⚠️ 설정 문제: {len(validation_results)}개 모델에서 경고")
                else:
                    print("   ✅ 모든 설정이 안전합니다")

            except Exception as e:
                print(f"   ❌ 설정 파일 읽기 실패: {e}")
        else:
            print("   ❌ 설정 파일 없음 (init 명령 실행 필요)")

        # 디렉토리 확인
        directories = {
            "data": "데이터셋",
            "optimization_results": "최적화 결과",
            "logs": "로그 파일"
        }

        for dir_name, description in directories.items():
            dir_path = Path(dir_name)
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*")))
                print(f"   ✅ {description}: {file_count}개 파일")
            else:
                print(f"   ❌ {description} 디렉토리 없음")

        # 하드웨어 정보 (IMPORTS_OK일 때만)
        if IMPORTS_OK:
            try:
                hardware_info = HardwareDetector.detect_hardware()
                print(f"   💻 시스템:")
                print(f"      CPU: {hardware_info['cpu_cores']}코어")
                print(f"      메모리: {hardware_info['available_memory']}/{hardware_info['total_memory']}GB")

                if hardware_info['cuda_available']:
                    for i in range(hardware_info['cuda_device_count']):
                        gpu_name = hardware_info.get(f'gpu_{i}_name', f'GPU {i}')
                        gpu_memory = hardware_info.get(f'gpu_{i}_memory', 0)
                        print(f"      {gpu_name}: {gpu_memory}GB")
                else:
                    print(f"      GPU: 사용 불가")

            except Exception as e:
                print(f"   ⚠️ 하드웨어 정보 확인 불가: {e}")

    def run_init_command(self, args):
        """init 명령어 실행"""
        print("🔧 안전한 시스템 초기화...")
        self._initialize()

        config_file = Path("llm_config.json")

        if config_file.exists() and not args.force:
            print(f"⚠️ 설정 파일이 이미 존재합니다: {config_file}")
            print("   덮어쓰려면 --force 옵션을 사용하세요.")
            return

        try:
            # 설정 매니저 생성
            self.config_manager = ConfigManager()
            print(f"✅ 안전한 설정 파일 생성: {config_file}")

            # 디렉토리 생성
            directories = ["data", "optimization_results", "logs"]
            for directory in directories:
                Path(directory).mkdir(exist_ok=True)
                print(f"   ✅ {directory} 디렉토리 생성")

            # 하드웨어 자동 감지
            if args.auto_detect:
                print("🔍 하드웨어 자동 감지...")
                hardware_info = HardwareDetector.detect_hardware()
                print(f"   GPU: {hardware_info['cuda_device_count']}개")
                print(f"   메모리: {hardware_info['total_memory']}GB")

                if hardware_info['cuda_available']:
                    total_gpu_memory = sum(
                        hardware_info.get(f'gpu_{i}_memory', 0)
                        for i in range(hardware_info['cuda_device_count'])
                    )
                    print(f"   GPU 메모리: {total_gpu_memory}GB")

            print("\n📝 다음 단계:")
            print("1. 하드웨어 정보 확인: python safe_main.py hardware")
            print("2. 모델 목록 확인: python safe_main.py list --type models")
            print(
                "3. 안전한 최적화 실행: python safe_main.py optimize --model qwen2.5-7b --dataset korean_math --samples 10 --safe")

        except Exception as e:
            if self.logger:
                self.logger.error(f"초기화 실패: {e}")
            print(f"❌ 초기화 실패: {e}")
            if args.debug:
                traceback.print_exc()

    def run_hardware_command(self, args):
        """hardware 명령어 실행"""
        print("💻 하드웨어 정보 분석")
        self._initialize()

        try:
            hardware_info = HardwareDetector.detect_hardware()

            print(f"\n🔍 감지된 하드웨어:")
            print(f"   플랫폼: {hardware_info['platform']}")
            print(f"   CUDA 사용 가능: {'✅' if hardware_info['cuda_available'] else '❌'}")
            print(f"   MPS 사용 가능: {'✅' if hardware_info.get('mps_available', False) else '❌'}")
            print(f"   GPU 개수: {hardware_info['cuda_device_count']}")
            print(f"   총 메모리: {hardware_info['total_memory']}GB")
            print(f"   사용 가능 메모리: {hardware_info['available_memory']}GB")
            print(f"   CPU 코어: {hardware_info['cpu_cores']}")

            if hardware_info['cuda_available']:
                total_gpu_memory = 0
                for i in range(hardware_info['cuda_device_count']):
                    gpu_memory = hardware_info.get(f'gpu_{i}_memory', 0)
                    gpu_name = hardware_info.get(f'gpu_{i}_name', 'Unknown')
                    compute_cap = hardware_info.get(f'gpu_{i}_compute_capability', 'Unknown')
                    print(f"   GPU {i}: {gpu_name} ({gpu_memory}GB, CC {compute_cap})")
                    total_gpu_memory += gpu_memory

                print(f"\n🎯 안전한 모델 추천:")
                if total_gpu_memory >= 80:
                    print("   ✅ 70B 모델까지 실행 가능 (4-bit 양자화 권장)")
                elif total_gpu_memory >= 32:
                    print("   ✅ 30B 모델까지 실행 가능 (4-bit 양자화 권장)")
                elif total_gpu_memory >= 16:
                    print("   ✅ 13B 모델까지 실행 가능 (4-bit 양자화 권장)")
                elif total_gpu_memory >= 8:
                    print("   ✅ 7B 모델 실행 가능 (4-bit 양자화 필수)")
                else:
                    print("   ⚠️ CPU 추론 권장 (작은 모델만)")

            # 모델 크기별 추천
            if args.model_size:
                print(f"\n🎯 {args.model_size.upper()} 모델 안전 설정:")
                try:
                    recommended = HardwareDetector.get_recommended_config(args.model_size, hardware_info)
                    print(f"   장치: {recommended.device}")
                    print(f"   데이터 타입: {recommended.dtype}")
                    print(f"   4-bit 양자화: {'✅' if recommended.load_in_4bit else '❌'}")
                    print(f"   8-bit 양자화: {'✅' if recommended.load_in_8bit else '❌'}")
                    if hasattr(recommended, 'cpu_offload'):
                        print(f"   CPU 오프로드: {'✅' if recommended.cpu_offload else '❌'}")
                except Exception as e:
                    print(f"   ❌ 추천 설정 생성 실패: {e}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"하드웨어 정보 확인 실패: {e}")
            print(f"❌ 하드웨어 정보 확인 실패: {e}")

    async def run_optimize_command(self, args):
        """optimize 명령어 실행"""
        print(f"🔧 안전한 파라미터 최적화: {args.model} on {args.dataset}")
        self._initialize()

        # 안전 모드 제한
        if args.safe:
            max_samples = min(args.samples, 10)
            max_trials = min(args.trials, 5)
            print(f"   🛡️ 안전 모드: 샘플 {max_samples}개, 시도 {max_trials}회")
        else:
            max_samples = min(args.samples, 50)
            max_trials = min(args.trials, 20)

        try:
            # 최적화기 초기화
            self.optimizer = SafeOptimizer()

            # 모델 설정 확인
            model_config = self.optimizer.config_manager.get_model_config(args.model)
            if not model_config:
                print(f"❌ 모델 {args.model} 설정을 찾을 수 없습니다.")
                print("   사용 가능한 모델:")
                for name in self.optimizer.config_manager.model_configs.keys():
                    print(f"     - {name}")
                return

            print(f"   📋 모델: {model_config.model_path}")
            print(f"   🎯 전략: {args.strategy}")
            print(f"   📊 평가자: {args.evaluator}")

            # 최적화 실행
            result = await self.optimizer.optimize_parameters(
                model_name=args.model,
                dataset_name=args.dataset,
                evaluator_type=args.evaluator,
                optimization_strategy=args.strategy,
                max_trials=max_trials,
                num_samples=max_samples,
                timeout_seconds=args.timeout
            )

            print(f"✅ 최적화 완료!")
            print(f"   최고 점수: {result.best_score:.3f}")
            print(f"   소요 시간: {result.total_time:.1f}초")

            print(f"\n🎯 최적 파라미터:")
            params = result.best_params
            print(f"   Temperature: {params.temperature:.3f}")
            print(f"   Top-p: {params.top_p:.3f}")
            print(f"   Top-k: {params.top_k}")
            print(f"   Max tokens: {params.max_new_tokens}")
            print(f"   Repetition penalty: {params.repetition_penalty:.3f}")

            if result.recommendations:
                print(f"\n💡 추천사항:")
                for i, rec in enumerate(result.recommendations, 1):
                    print(f"   {i}. {rec}")

            print(f"\n📁 결과 저장: optimization_results/{result.test_id}.json")

        except Exception as e:
            if self.logger:
                self.logger.error(f"최적화 실패: {e}")
            print(f"❌ 최적화 실패: {e}")

            # 구체적인 해결책 제시
            if "CUDA out of memory" in str(e):
                print("💡 해결 방법:")
                print("   1. --samples 수를 줄이세요 (예: --samples 5)")
                print("   2. --safe 옵션을 사용하세요")
                print("   3. 더 작은 모델을 사용하세요")
            elif "not found" in str(e):
                print("💡 해결 방법:")
                print("   1. python safe_main.py list --type models (모델 목록 확인)")
                print("   2. python safe_main.py init --auto-detect (설정 초기화)")

            if args.debug:
                traceback.print_exc()

    async def run_benchmark_command(self, args):
        """benchmark 명령어 실행"""
        print(f"⚡ 안전한 벤치마크: {args.model} on {args.dataset}")
        self._initialize()

        # 안전한 제한
        max_samples = min(args.samples, 50)
        max_iterations = min(args.iterations, 3)

        try:
            self.optimizer = SafeOptimizer()

            # 파라미터 생성
            from config import InferenceParams
            params = InferenceParams(
                temperature=args.temperature,
                top_p=getattr(args, 'top_p', 0.9),  # args.top_p 대신 안전한 접근
                max_new_tokens=min(args.max_tokens, 512)
            )

            print(f"   📊 샘플: {max_samples}개")
            print(f"   🔄 반복: {max_iterations}회")

            # 벤치마크 실행
            result = await self.optimizer.benchmark_model(
                model_name=args.model,
                dataset_name=args.dataset,
                params=params,
                num_samples=max_samples,
                iterations=max_iterations
            )

            print(f"✅ 벤치마크 완료!")

            # 성능 메트릭 출력
            perf = result.performance_metrics
            print(f"\n📊 성능 메트릭:")
            print(f"   토큰/초: {perf.get('tokens_per_second', 0):.1f}")
            print(f"   평균 지연시간: {perf.get('latency_avg', 0):.3f}초")
            print(f"   P95 지연시간: {perf.get('latency_p95', 0):.3f}초")
            print(f"   메모리 사용량: {perf.get('memory_usage_mb', 0):.0f}MB")
            print(f"   처리량: {perf.get('throughput', 0):.1f} req/sec")

            # 정확도
            accuracy = perf.get('accuracy', 0)
            print(f"   정확도: {accuracy:.3f}")

            # 비용 분석
            if result.cost_analysis:
                cost = result.cost_analysis
                print(f"\n💰 비용 분석:")
                print(f"   시간당 비용: ${cost.get('cost_per_hour_usd', 0):.4f}")
                print(f"   1K토큰당 비용: ${cost.get('cost_per_1k_tokens_usd', 0):.6f}")

            print(f"\n📁 결과 저장: optimization_results/bench_{result.test_id}.json")

        except Exception as e:
            if self.logger:
                self.logger.error(f"벤치마크 실패: {e}")
            print(f"❌ 벤치마크 실패: {e}")
            if args.debug:
                traceback.print_exc()

    async def run_compare_command(self, args):
        """compare 명령어 실행"""
        print(f"⚖️ 안전한 모델 비교: {', '.join(args.models)} on {args.dataset}")
        self._initialize()

        max_samples = min(args.samples, 30)

        try:
            self.optimizer = SafeOptimizer()

            # 기본 파라미터
            from config import InferenceParams
            params = InferenceParams(temperature=0.1, top_p=0.9, max_new_tokens=200)

            results = {}
            for model in args.models:
                print(f"\n🔄 {model} 테스트 중...")

                try:
                    result = await self.optimizer.benchmark_model(
                        model_name=model,
                        dataset_name=args.dataset,
                        params=params,
                        num_samples=max_samples,
                        iterations=1
                    )

                    results[model] = result
                    perf = result.performance_metrics
                    accuracy = perf.get('accuracy', 0)
                    speed = perf.get('tokens_per_second', 0)
                    print(f"   ✅ 완료: 정확도 {accuracy:.3f}, {speed:.1f} tokens/sec")

                except Exception as e:
                    print(f"   ❌ 실패: {e}")
                    continue

            # 결과 정렬 및 출력
            if results:
                print(f"\n📊 비교 결과 ({args.metric} 기준):")
                print(f"{'순위':<4} {'모델':<20} {'정확도':<8} {'토큰/초':<10} {'메모리(MB)':<12}")
                print("-" * 60)

                # 정렬
                if args.metric == 'accuracy':
                    sorted_results = sorted(results.items(),
                                            key=lambda x: x[1].performance_metrics.get('accuracy', 0),
                                            reverse=True)
                elif args.metric == 'speed':
                    sorted_results = sorted(results.items(),
                                            key=lambda x: x[1].performance_metrics.get('tokens_per_second', 0),
                                            reverse=True)
                else:  # efficiency
                    sorted_results = sorted(results.items(),
                                            key=lambda x: x[1].hardware_efficiency.get('overall_efficiency', 0),
                                            reverse=True)

                for i, (model, result) in enumerate(sorted_results, 1):
                    perf = result.performance_metrics
                    accuracy = perf.get('accuracy', 0)
                    speed = perf.get('tokens_per_second', 0)
                    memory = perf.get('memory_usage_mb', 0)

                    rank_symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    print(f"{rank_symbol:<4} {model:<20} {accuracy:<8.3f} {speed:<10.1f} {memory:<12.0f}")
            else:
                print("❌ 비교할 결과가 없습니다.")

        except Exception as e:
            if self.logger:
                self.logger.error(f"모델 비교 실패: {e}")
            print(f"❌ 모델 비교 실패: {e}")
            if args.debug:
                traceback.print_exc()

    def run_list_command(self, args):
        """list 명령어 실행"""
        try:
            self._initialize()

            if not self.config_manager:
                self.config_manager = ConfigManager()

            if args.type == 'models':
                models = self.config_manager.model_configs
                print(f"📋 등록된 모델 ({len(models)}개):")

                for name, config in models.items():
                    from config import SafetyChecker
                    warnings = SafetyChecker.check_model_config(config)
                    safety_status = "⚠️" if warnings else "✅"

                    print(f"   {safety_status} {name}")
                    print(f"      경로: {config.model_path}")
                    print(f"      유형: {config.model_type}")
                    print(f"      장치: {config.device}")
                    print(f"      양자화: 4bit={config.load_in_4bit}, 8bit={config.load_in_8bit}")
                    if warnings:
                        print(f"      경고: {len(warnings)}개")
                    print()

            elif args.type == 'datasets':
                datasets = self.config_manager.test_configs
                print(f"📋 등록된 데이터셋 ({len(datasets)}개):")

                for name, config in datasets.items():
                    data_file = Path(config.dataset_path)
                    exists = "✅" if data_file.exists() else "❌"
                    print(f"   {exists} {name}")
                    print(f"      경로: {config.dataset_path}")
                    print(f"      샘플: {config.num_samples}개")
                    print()

            elif args.type == 'results':
                results_dir = Path("optimization_results")
                if results_dir.exists():
                    result_files = list(results_dir.glob("*.json"))
                    print(f"📋 저장된 결과 ({len(result_files)}개):")

                    for result_file in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                        try:
                            with open(result_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)

                            test_type = "🔧" if 'best_score' in data else "⚡"
                            model = data.get('model_name', 'Unknown')
                            dataset = data.get('dataset_name', 'Unknown')
                            timestamp = data.get('timestamp', '')[:16].replace('T', ' ')

                            print(f"   {test_type} {result_file.name}")
                            print(f"      {model} on {dataset} ({timestamp})")

                        except Exception:
                            print(f"   ❌ {result_file.name} (읽기 실패)")
                else:
                    print("📋 저장된 결과가 없습니다.")

        except Exception as e:
            if self.logger:
                self.logger.error(f"정보 조회 실패: {e}")
            print(f"❌ 정보 조회 실패: {e}")

    def run_status_command(self, args):
        """status 명령어 실행"""
        print("🔍 시스템 상태 점검")
        self.show_system_status()

        # 추가 상태 정보
        if IMPORTS_OK:
            try:
                from config import get_resource_manager
                resource_manager = get_resource_manager()
                memory_usage = resource_manager.get_memory_usage()

                if memory_usage:
                    print(f"\n🎯 현재 리소스 사용량:")
                    for key, value in memory_usage.items():
                        if isinstance(value, float) and 'percent' in key:
                            status = "⚠️" if value > 80 else "✅"
                            print(f"   {status} {key}: {value:.1f}%")
                        elif isinstance(value, float) and 'gb' in key:
                            print(f"      {key}: {value:.2f}GB")

            except Exception as e:
                print(f"⚠️ 리소스 정보 확인 실패: {e}")

    def run_clean_command(self, args):
        """clean 명령어 실행"""
        print("🧹 시스템 정리 시작...")

        cleaned_items = []

        try:
            if args.cache or args.all:
                # CUDA 캐시 정리
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        cleaned_items.append("CUDA 캐시")
                except:
                    pass

                # Python 가비지 컬렉션
                gc.collect()
                cleaned_items.append("Python 가비지")

            if args.logs or args.all:
                # 오래된 로그 파일 정리 (7일 이상)
                log_dir = Path("logs")
                if log_dir.exists():
                    current_time = time.time()
                    old_logs = []

                    for log_file in log_dir.glob("*.log"):
                        if current_time - log_file.stat().st_mtime > 7 * 24 * 3600:  # 7일
                            old_logs.append(log_file)

                    for log_file in old_logs:
                        log_file.unlink()

                    if old_logs:
                        cleaned_items.append(f"{len(old_logs)}개 오래된 로그 파일")

            if args.results or args.all:
                # 사용자 확인 후 결과 정리
                results_dir = Path("optimization_results")
                if results_dir.exists():
                    result_files = list(results_dir.glob("*.json"))
                    if result_files:
                        response = input(f"⚠️ {len(result_files)}개 결과 파일을 정리하시겠습니까? (y/N): ")
                        if response.lower() == 'y':
                            for result_file in result_files:
                                result_file.unlink()
                            cleaned_items.append(f"{len(result_files)}개 결과 파일")

            # HuggingFace 캐시 정리 (선택적)
            if args.all:
                hf_cache = Path.home() / ".cache" / "huggingface"
                if hf_cache.exists():
                    response = input("⚠️ HuggingFace 캐시를 정리하시겠습니까? (y/N): ")
                    if response.lower() == 'y':
                        shutil.rmtree(hf_cache, ignore_errors=True)
                        cleaned_items.append("HuggingFace 캐시")

            if cleaned_items:
                print("✅ 정리 완료:")
                for item in cleaned_items:
                    print(f"   - {item}")
            else:
                print("✅ 정리할 항목이 없습니다.")

        except Exception as e:
            if self.logger:
                self.logger.error(f"정리 작업 실패: {e}")
            print(f"❌ 정리 작업 실패: {e}")

    def show_welcome_screen(self):
        """환영 화면"""
        self.print_banner()
        print("🛡️ 안전한 오픈소스 LLM 추론 성능 최적화 시스템에 오신 것을 환영합니다!")

        print("\n🚀 빠른 시작 (안전 모드):")
        print("1. 시스템 초기화: python safe_main.py init --auto-detect")
        print("2. 하드웨어 확인: python safe_main.py hardware")
        print("3. 시스템 상태: python safe_main.py status")
        print("4. 안전한 최적화: python safe_main.py optimize --model qwen2.5-7b --dataset korean_math --samples 10 --safe")
        print("5. 성능 벤치마크: python safe_main.py benchmark --model qwen2.5-7b --dataset korean_qa --samples 20")

        print("\n🔧 주요 개선사항:")
        print("   ✅ 메모리 누수 방지 - 자동 리소스 정리")
        print("   ✅ 스레드 안전성 - 동시성 문제 해결")
        print("   ✅ 의존성 안정성 - Optuna 기반 최적화")
        print("   ✅ 오류 복구 - 강화된 예외 처리")
        print("   ✅ 보안 강화 - 입력 검증 및 안전 모드")

        print("\n⚡ 안전 사용 팁:")
        print("   🛡️ 처음 사용: --safe 옵션 필수")
        print("   💾 메모리 절약: --samples 10-20 권장")
        print("   🐛 문제 해결: --debug 옵션 활용")
        print("   🔧 시스템 정리: clean 명령어 정기 실행")

        print("\n💡 도움말:")
        print("   전체 명령어: python safe_main.py --help")
        print("   명령어별 도움말: python safe_main.py [명령어] --help")

        # 시스템 상태 간단히 표시
        self.show_system_status()

    async def main(self):
        """메인 함수"""
        parser = self.create_argument_parser()

        # 인자가 없으면 환영 화면
        if len(sys.argv) == 1:
            self.show_welcome_screen()
            return

        try:
            args = parser.parse_args()
        except SystemExit:
            return

        # 로깅 설정
        setup_safe_logging(args.debug)

        if not args.command:
            self.print_banner()
            parser.print_help()
            return

        self.print_banner()

        # 독립적 명령어들 (시스템 요구사항 확인 불필요)
        if args.command in ['init', 'hardware', 'list', 'status', 'clean']:
            try:
                if args.command == 'init':
                    self.run_init_command(args)
                elif args.command == 'hardware':
                    self.run_hardware_command(args)
                elif args.command == 'list':
                    self.run_list_command(args)
                elif args.command == 'status':
                    self.run_status_command(args)
                elif args.command == 'clean':
                    self.run_clean_command(args)
                return
            except KeyboardInterrupt:
                print("\n⏹️ 사용자에 의해 중단되었습니다.")
                return
            except Exception as e:
                if self.logger:
                    self.logger.error(f"명령 실행 실패: {e}")
                print(f"❌ 명령 실행 실패: {e}")
                if args.debug:
                    traceback.print_exc()
                return

        # 복잡한 명령어들 (시스템 요구사항 확인 필요)
        if not self.check_system_requirements():
            print("\n⚠️ 시스템 요구사항을 만족하지 않습니다.")
            if not args.safe:
                response = input("안전 모드로 계속 진행하시겠습니까? (y/N): ")
                if response.lower() != 'y':
                    return
                args.safe = True

        try:
            # 타임아웃 설정
            tasks = []

            # 명령 실행
            if args.command == 'optimize':
                command_task = asyncio.create_task(self.run_optimize_command(args))
            elif args.command == 'benchmark':
                command_task = asyncio.create_task(self.run_benchmark_command(args))
            elif args.command == 'compare':
                command_task = asyncio.create_task(self.run_compare_command(args))
            else:
                print(f"❌ 알 수 없는 명령어: {args.command}")
                return

            tasks.append(command_task)

            # 타임아웃 태스크 (필요한 경우)
            if args.timeout > 0:
                timeout_task = asyncio.create_task(asyncio.sleep(args.timeout))
                tasks.append(timeout_task)

            # 태스크 실행
            if len(tasks) > 1:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                # 타임아웃 체크
                if timeout_task in done:
                    command_task.cancel()
                    print(f"\n⏰ 시간 초과 ({args.timeout}초)")
                else:
                    timeout_task.cancel()
            else:
                await command_task

        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중단되었습니다.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"실행 중 오류: {e}")
            print(f"❌ 실행 중 오류: {e}")
            if args.debug:
                traceback.print_exc()
        finally:
            # 정리 작업
            print("\n🧹 시스템 정리 중...")
            try:
                if self.optimizer:
                    # 최적화기 정리는 자동으로 처리됨
                    pass
                cleanup_all()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"정리 중 오류: {e}")
            print("✅ 정리 완료")


def main():
    """진입점"""
    # 기본 오류 처리
    if not IMPORTS_OK:
        print("❌ 필수 모듈을 import할 수 없습니다.")
        print(f"오류: {IMPORT_ERROR}")
        print("\n📦 다음 명령으로 의존성을 설치하세요:")
        print("   pip install torch transformers numpy pandas optuna psutil")
        sys.exit(1)

    try:
        cli = SafeCLI()
        asyncio.run(cli.main())
    except KeyboardInterrupt:
        print("\n⏹️ 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
        print("🛡️ 안전 모드로 다시 시도해보세요: --safe")
        print("🐛 문제 지속 시 --debug 옵션으로 상세 정보 확인")
    finally:
        # 최종 정리
        cleanup_all()


if __name__ == "__main__":
    main()


    def create_argument_parser(self) -> argparse.ArgumentParser:
        """CLI 인자 파서 생성"""
        parser = argparse.ArgumentParser(
            description='안전한 오픈소스 LLM 추론 성능 최적화 시스템',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
안전한 사용 예시:
  python safe_main.py init --auto-detect              # 시스템 초기화
  python safe_main.py hardware                        # 하드웨어 정보 확인
  python safe_main.py optimize --model qwen2.5-7b --dataset korean_math --samples 20 --safe
  python safe_main.py benchmark --model qwen2.5-7b --dataset korean_qa --samples 30
  python safe_main.py compare --models qwen2.5-7b llama2-7b --dataset korean_math
  python safe_main.py status                          # 시스템 상태 확인

안전 모드 옵션:
  --safe: 메모리와 성능을 제한하여 안전하게 실행
  --debug: 상세한 디버그 정보 출력
  --timeout: 작업 시간 제한 (초)
            """
        )

        # 전역 옵션
        parser.add_argument('--safe', action='store_true', help='안전 모드로 실행')
        parser.add_argument('--debug', action='store_true', help='디버그 모드')
        parser.add_argument('--timeout', type=int, default=3600, help='작업 시간 제한 (초)')

        subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')

        # init 명령어
        init_parser = subparsers.add_parser('init', help='시스템 초기화')
        init_parser.add_argument('--force', action='store_true', help='기존 설정 덮어쓰기')
        init_parser.add_argument('--auto-detect', action='store_true', help='하드웨어 자동 감지')

        # hardware 명령어
        hardware_parser = subparsers.add_parser('hardware', help='하드웨어 정보 확인')
        hardware_parser.add_argument('--model-size', choices=['1b', '3b', '7b', '13b', '30b', '70b'],
                                     help='모델 크기별 추천')

        # optimize 명령어
        optimize_parser = subparsers.add_parser('optimize', help='파라미터 최적화')
        optimize_parser.add_argument('--model', required=True, help='모델 이름')
        optimize_parser.add_argument('--dataset', required=True, help='데이터셋 이름')
        optimize_parser.add_argument('--strategy', choices=['optuna', 'grid_search'],
                                     default='grid_search', help='최적화 전략')
        optimize_parser.add_argument('--trials', type=int, default=10, help='최적화 시도 횟수')
        optimize_parser.add_argument('--samples', type=int, default=20, help='테스트 샘플 수')
        optimize_parser.add_argument('--evaluator', choices=['exact_match', 'contains', 'similarity'],
                                     default='similarity', help='평가자 유형')

        # benchmark 명령어
        benchmark_parser = subparsers.add_parser('benchmark', help='성능 벤치마크')
        benchmark_parser.add_argument('--model', required=True, help='모델 이름')
        benchmark_parser.add_argument('--dataset', required=True, help='데이터셋 이름')
        benchmark_parser.add_argument('--samples', type=int, default=30, help='테스트 샘플 수')
        benchmark_parser.add_argument('--iterations', type=int, default=1, help='반복 횟수')
        benchmark_parser.add_argument('--temperature', type=float, default=0.1, help='Temperature')
        benchmark_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p')
        benchmark_parser.add_argument('--max-tokens', type=int, default=200, help='최대 토큰 수')

        # compare 명령어
        compare_parser = subparsers.add_parser('compare', help='모델 비교')
        compare_parser.add_argument('--models', nargs='+', required=True, help='비교할 모델들')
        compare_parser.add_argument('--dataset', required=True, help='데이터셋 이름')
        compare_parser.add_argument('--samples', type=int, default=20, help='테스트 샘플 수')
        compare_parser.add_argument('--metric', choices=['accuracy', 'speed', 'efficiency'],
                                    default='accuracy', help='비교 기준')

        # list 명령어
        list_parser = subparsers.add_parser('list', help='정보 조회')
        list_parser.add_argument('--type', choices=['models', 'datasets', 'results'],
                                 default='models', help='조회할 정보 유형')

        # status 명령어
        status_parser = subparsers.add_parser('status', help='시스템 상태 확인')

        # clean 명령어
        clean_parser = subparsers.add_parser('clean', help='시스템 정리')
        clean_parser.add_argument('--cache', action='store_true', help='캐시 정리')
        clean_parser.add_argument('--logs', action='store_true', help='로그 정리')
        clean_parser.add_argument('--results', action='store_true', help='결과 정리')
        clean_parser.add_argument('--all', action='store_true', help='모든 정리')

        return parser


    def print_banner(self):
        """시스템 배너 출력"""
        banner = """
╭─────────────────────────────────────────────────────────────╮
│          🛡️ 안전한 오픈소스 LLM 추론 성능 최적화 시스템         │
│              Safe Open Source LLM Optimizer               │
│                    ✅ Security Enhanced                    │
╰─────────────────────────────────────────────────────────────╯
"""
        print(banner)


    def check_system_requirements(self) -> bool:
        """시스템 요구사항 확인"""
        requirements_met = True
        issues = []

        # Python 버전 확인
        if sys.version_info < (3, 8):
            issues.append(f"Python 3.8+ 필요 (현재: {sys.version})")
            requirements_met = False

        # 필수 패키지 확인
        required_packages = {
            'torch': 'PyTorch',
            'transformers': 'HuggingFace Transformers',
            'numpy': 'NumPy',
            'psutil': 'psutil'
        }

        missing_packages = []
        for package, description in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(f"{package} ({description})")

        if missing_packages:
            issues.append(f"누락된 필수 패키지: {', '.join(missing_packages)}")
            requirements_met = False

        # 선택적 패키지 확인
        optional_packages = {
            'optuna': 'Optuna (고급 최적화용)',
            'plotly': 'Plotly (시각화용)'
        }

        missing_optional = []
        for package, description in optional_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(f"{package} ({description})")

        if missing_optional:
            print(f"⚠️ 선택적 패키지 누락: {', '.join(missing_optional)}")
            print("  일부 기능이 제한될 수 있습니다.")

        # 하드웨어 정보
        try:
            hardware_info = HardwareDetector.detect_hardware()
            if hardware_info['cuda_available']:
                gpu_count = hardware_info['cuda_device_count']
                total_memory = sum(
                    hardware_info.get(f'gpu_{i}_memory', 0)
                    for i in range(gpu_count)
                )
                print(f"✅ GPU 감지: {gpu_count}개, 총 메모리: {total_memory}GB")
            else:
                print("⚠️ CUDA 사용 불가 - CPU 모드로 실행됩니다.")
        except Exception as e:
            print(f"⚠️ 하드웨어 정보 확인 불가: {e}")

        # 오류 출력
        if issues:
            print(f"❌ 시스템 요구사항 미충족:")
            for issue in issues:
                print(f"   - {issue}")

            print(f"\n💡 해결 방법:")
            if missing_packages:
                packages = [p.split(' ')[0] for p in missing_packages]
                print(f"   pip install {' '.join(packages)}")

        return requirements_met


    def show_system_status(self):
        """시스템 상태 표시"""
        print("🔧 시스템 상태:")

        # 설정 파일 확인
        config_file = Path("llm_config.json")
        if config_file.exists():
            try:
                self.config_manager = ConfigManager()
                model_count = len(self.config_manager.model_configs)
                print(f"   ✅ 설정 파일: {model_count}개 모델 등록됨")

                # 설정 검증
                validation_results = self.config_manager.validate_all_configs()
                if validation_results:
                    print(f"   ⚠️ 설정 문제: {len(validation_results)}개 모델에서 경고")
                else:
                    print("   ✅ 모든 설정이 안전합니다")

            except Exception as e:
                print(f"   ❌ 설정 파일 읽기 실패: {e}")
        else:
            print("   ❌ 설정 파일 없음 (init 명령 실행 필요)")

        # 디렉토리 확인
        directories = {
            "data": "데이터셋",
            "optimization_results": "최적화 결과",
            "logs": "로그 파일"
        }

        for dir_name, description in directories.items():
            dir_path = Path(dir_name)
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*")))
                print(f"   ✅ {description}: {file_count}개 파일")
            else:
                print(f"   ❌ {description} 디렉토리 없음")

        # 하드웨어 정보
        try:
            hardware_info = HardwareDetector.detect_hardware()
            print(f"   💻 시스템:")
            print(f"      CPU: {hardware_info['cpu_cores']}코어")
            print(f"      메모리: {hardware_info['available_memory']}/{hardware_info['total_memory']}GB")

            if hardware_info['cuda_available']:
                for i in range(hardware_info['cuda_device_count']):
                    gpu_name = hardware_info.get(f'gpu_{i}_name', f'GPU {i}')
                    gpu_memory = hardware_info.get(f'gpu_{i}_memory', 0)
                    print(f"      {gpu_name}: {gpu_memory}GB")
            else:
                print(f"      GPU: 사용 불가")

        except Exception as e:
            print(f"   ⚠️ 하드웨어 정보 확인 불가: {e}")


    def run_init_command(self, args):
        """init 명령어 실행"""
        print("🔧 안전한 시스템 초기화...")

        config_file = Path("llm_config.json")

        if config_file.exists() and not args.force:
            print(f"⚠️ 설정 파일이 이미 존재합니다: {config_file}")
            print("   덮어쓰려면 --force 옵션을 사용하세요.")
            return

        try:
            # 설정 매니저 생성
            self.config_manager = ConfigManager()
            print(f"✅ 안전한 설정 파일 생성: {config_file}")

            # 디렉토리 생성
            directories = ["data", "optimization_results", "logs"]
            for directory in directories:
                Path(directory).mkdir(exist_ok=True)
                print(f"   ✅ {directory} 디렉토리 생성")

            # 하드웨어 자동 감지
            if args.auto_detect:
                print("🔍 하드웨어 자동 감지...")
                hardware_info = HardwareDetector.detect_hardware()
                print(f"   GPU: {hardware_info['cuda_device_count']}개")
                print(f"   메모리: {hardware_info['total_memory']}GB")

                if hardware_info['cuda_available']:
                    total_gpu_memory = sum(
                        hardware_info.get(f'gpu_{i}_memory', 0)
                        for i in range(hardware_info['cuda_device_count'])
                    )
                    print(f"   GPU 메모리: {total_gpu_memory}GB")

            print("\n📝 다음 단계:")
            print("1. 하드웨어 정보 확인: python safe_main.py hardware")
            print("2. 모델 목록 확인: python safe_main.py list --type models")
            print(
                "3. 안전한 최적화 실행: python safe_main.py optimize --model qwen2.5-7b --dataset korean_math --samples 10 --safe")

        except Exception as e:
            self.logger.error(f"초기화 실패: {e}")
            print(f"❌ 초기화 실패: {e}")
            if args.debug:
                traceback.print_exc()


    def run_hardware_command(self, args):
        """hardware 명령어 실행"""
        print("💻 하드웨어 정보 분석")

        try:
            hardware_info = HardwareDetector.detect_hardware()

            print(f"\n🔍 감지된 하드웨어:")
            print(f"   플랫폼: {hardware_info['platform']}")
            print(f"   CUDA 사용 가능: {'✅' if hardware_info['cuda_available'] else '❌'}")
            print(f"   MPS 사용 가능: {'✅' if hardware_info.get('mps_available', False) else '❌'}")
            print(f"   GPU 개수: {hardware_info['cuda_device_count']}")
            print(f"   총 메모리: {hardware_info['total_memory']}GB")
            print(f"   사용 가능 메모리: {hardware_info['available_memory']}GB")
            print(f"   CPU 코어: {hardware_info['cpu_cores']}")

            if hardware_info['cuda_available']:
                total_gpu_memory = 0
                for i in range(hardware_info['cuda_device_count']):
                    gpu_memory = hardware_info.get(f'gpu_{i}_memory', 0)
                    gpu_name = hardware_info.get(f'gpu_{i}_name', 'Unknown')
                    compute_cap = hardware_info.get(f'gpu_{i}_compute_capability', 'Unknown')
                    print(f"   GPU {i}: {gpu_name} ({gpu_memory}GB, CC {compute_cap})")
                    total_gpu_memory += gpu_memory

                print(f"\n🎯 안전한 모델 추천:")
                if total_gpu_memory >= 80:
                    print("   ✅ 70B 모델까지 실행 가능 (4-bit 양자화 권장)")
                elif total_gpu_memory >= 32:
                    print("   ✅ 30B 모델까지 실행 가능 (4-bit 양자화 권장)")
                elif total_gpu_memory >= 16:
                    print("   ✅ 13B 모델까지 실행 가능 (4-bit 양자화 권장)")
                elif total_gpu_memory >= 8:
                    print("   ✅ 7B 모델 실행 가능 (4-bit 양자화 필수)")
                else:
                    print("   ⚠️ CPU 추론 권장 (작은 모델만)")

            # 모델 크기별 추천
            if args.model_size:
                print(f"\n🎯 {args.model_size.upper()} 모델 안전 설정:")
                try:
                    recommended = HardwareDetector.get_recommended_config(args.model_size, hardware_info)
                    print(f"   장치: {recommended.device}")
                    print(f"   데이터 타입: {recommended.dtype}")
                    print(f"   4-bit 양자화: {'✅' if recommended.load_in_4bit else '❌'}")
                    print(f"   8-bit 양자화: {'✅' if recommended.load_in_8bit else '❌'}")
                    print(f"   CPU 오프로드: {'✅' if recommended.cpu_offload else '❌'}")
                except Exception as e:
                    print(f"   ❌ 추천 설정 생성 실패: {e}")

        except Exception as e:
            self.logger.error(f"하드웨어 정보 확인 실패: {e}")
            print(f"❌ 하드웨어 정보 확인 실패: {e}")


    async def run_optimize_command(self, args):
        """optimize 명령어 실행"""
        print(f"🔧 안전한 파라미터 최적화: {args.model} on {args.dataset}")

        # 안전 모드 제한
        if args.safe:
            max_samples = min(args.samples, 10)
            max_trials = min(args.trials, 5)
            print(f"   🛡️ 안전 모드: 샘플 {max_samples}개, 시도 {max_trials}회")
        else:
            max_samples = min(args.samples, 50)
            max_trials = min(args.trials, 20)

        try:
            # 최적화기 초기화
            self.optimizer = SafeOptimizer()

            # 모델 설정 확인
            model_config = self.optimizer.config_manager.get_model_config(args.model)
            if not model_config:
                print(f"❌ 모델 {args.model} 설정을 찾을 수 없습니다.")
                print("   사용 가능한 모델:")
                for name in self.optimizer.config_manager.model_configs.keys():
                    print(f"     - {name}")
                return

            print(f"   📋 모델: {model_config.model_path}")
            print(f"   🎯 전략: {args.strategy}")
            print(f"   📊 평가자: {args.evaluator}")

            # 최적화 실행
            result = await self.optimizer.optimize_parameters(
                model_name=args.model,
                dataset_name=args.dataset,
                evaluator_type=args.evaluator,
                optimization_strategy=args.strategy,
                max_trials=max_trials,
                num_samples=max_samples,
                timeout_seconds=args.timeout
            )

            print(f"✅ 최적화 완료!")
            print(f"   최고 점수: {result.best_score:.3f}")
            print(f"   소요 시간: {result.total_time:.1f}초")

            print(f"\n🎯 최적 파라미터:")
            params = result.best_params
            print(f"   Temperature: {params.temperature:.3f}")
            print(f"   Top-p: {params.top_p:.3f}")
            print(f"   Top-k: {params.top_k}")
            print(f"   Max tokens: {params.max_new_tokens}")
            print(f"   Repetition penalty: {params.repetition_penalty:.3f}")

            if result.recommendations:
                print(f"\n💡 추천사항:")
                for i, rec in enumerate(result.recommendations, 1):
                    print(f"   {i}. {rec}")

            print(f"\n📁 결과 저장: optimization_results/{result.test_id}.json")

        except Exception as e:
            self.logger.error(f"최적화 실패: {e}")
            print(f"❌ 최적화 실패: {e}")

            # 구체적인 해결책 제시
            if "CUDA out of memory" in str(e):
                print("💡 해결 방법:")
                print("   1. --samples 수를 줄이세요 (예: --samples 5)")
                print("   2. --safe 옵션을 사용하세요")
                print("   3. 더 작은 모델을 사용하세요")
            elif "not found" in str(e):
                print("💡 해결 방법:")
                print("   1. python safe_main.py list --type models (모델 목록 확인)")
                print("   2. python safe_main.py init --auto-detect (설정 초기화)")

            if args.debug:
                traceback.print_exc()


    async def run_benchmark_command(self, args):
        """benchmark 명령어 실행"""
        print(f"⚡ 안전한 벤치마크: {args.model} on {args.dataset}")

        # 안전한 제한
        max_samples = min(args.samples, 50)
        max_iterations = min(args.iterations, 3)

        try:
            self.optimizer = SafeOptimizer()

            # 파라미터 생성
            from config import InferenceParams
            params = InferenceParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=min(args.max_tokens, 512)
            )

            print(f"   📊 샘플: {max_samples}개")
            print(f"   🔄 반복: {max_iterations}회")

            # 벤치마크 실행
            result = await self.optimizer.benchmark_model(
                model_name=args.model,
                dataset_name=args.dataset,
                params=params,
                num_samples=max_samples,
                iterations=max_iterations
            )

            print(f"✅ 벤치마크 완료!")

            # 성능 메트릭 출력
            perf = result.performance_metrics
            print(f"\n📊 성능 메트릭:")
            print(f"   토큰/초: {perf.get('tokens_per_second', 0):.1f}")
            print(f"   평균 지연시간: {perf.get('latency_avg', 0):.3f}초")
            print(f"   P95 지연시간: {perf.get('latency_p95', 0):.3f}초")
            print(f"   메모리 사용량: {perf.get('memory_usage_mb', 0):.0f}MB")
            print(f"   처리량: {perf.get('throughput', 0):.1f} req/sec")

            # 정확도
            accuracy = perf.get('accuracy', 0)
            print(f"   정확도: {accuracy:.3f}")

            # 비용 분석
            if result.cost_analysis:
                cost = result.cost_analysis
                print(f"\n💰 비용 분석:")
                print(f"   시간당 비용: ${cost.get('cost_per_hour_usd', 0):.4f}")
                print(f"   1K토큰당 비용: ${cost.get('cost_per_1k_tokens_usd', 0):.6f}")

            print(f"\n📁 결과 저장: optimization_results/bench_{result.test_id}.json")

        except Exception as e:
            self.logger.error(f"벤치마크 실패: {e}")
            print(f"❌ 벤치마크 실패: {e}")
            if args.debug:
                traceback.print_exc()


    async def run_compare_command(self, args):
        """compare 명령어 실행"""
        print(f"⚖️ 안전한 모델 비교: {', '.join(args.models)} on {args.dataset}")

        max_samples = min(args.samples, 30)

        try:
            self.optimizer = SafeOptimizer()

            # 기본 파라미터
            from config import InferenceParams
            params = InferenceParams(temperature=0.1, top_p=0.9, max_new_tokens=200)

            results = {}
            for model in args.models:
                print(f"\n🔄 {model} 테스트 중...")

                try:
                    result = await self.optimizer.benchmark_model(
                        model_name=model,
                        dataset_name=args.dataset,
                        params=params,
                        num_samples=max_samples,
                        iterations=1
                    )

                    results[model] = result
                    perf = result.performance_metrics
                    accuracy = perf.get('accuracy', 0)
                    speed = perf.get('tokens_per_second', 0)
                    print(f"   ✅ 완료: 정확도 {accuracy:.3f}, {speed:.1f} tokens/sec")

                except Exception as e:
                    print(f"   ❌ 실패: {e}")
                    continue

            # 결과 정렬 및 출력
            if results:
                print(f"\n📊 비교 결과 ({args.metric} 기준):")
                print(f"{'순위':<4} {'모델':<20} {'정확도':<8} {'토큰/초':<10} {'메모리(MB)':<12}")
                print("-" * 60)

                # 정렬
                if args.metric == 'accuracy':
                    sorted_results = sorted(results.items(),
                                            key=lambda x: x[1].performance_metrics.get('accuracy', 0),
                                            reverse=True)
                elif args.metric == 'speed':
                    sorted_results = sorted(results.items(),
                                            key=lambda x: x[1].performance_metrics.get('tokens_per_second', 0),
                                            reverse=True)
                else:  # efficiency
                    sorted_results = sorted(results.items(),
                                            key=lambda x: x[1].hardware_efficiency.get('overall_efficiency', 0),
                                            reverse=True)

                for i, (model, result) in enumerate(sorted_results, 1):
                    perf = result.performance_metrics
                    accuracy = perf.get('accuracy', 0)
                    speed = perf.get('tokens_per_second', 0)
                    memory = perf.get('memory_usage_mb', 0)

                    rank_symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    print(f"{rank_symbol:<4} {model:<20} {accuracy:<8.3f} {speed:<10.1f} {memory:<12.0f}")
            else:
                print("❌ 비교할 결과가 없습니다.")

        except Exception as e:
            self.logger.error(f"모델 비교 실패: {e}")
            print(f"❌ 모델 비교 실패: {e}")
            if args.debug:
                traceback.print_exc()


    def run_list_command(self, args):
        """list 명령어 실행"""
        try:
            if not self.config_manager:
                self.config_manager = ConfigManager()

            if args.type == 'models':
                models = self.config_manager.model_configs
                print(f"📋 등록된 모델 ({len(models)}개):")

                for name, config in models.items():
                    from config import SafetyChecker
                    warnings = SafetyChecker.check_model_config(config)
                    safety_status = "⚠️" if warnings else "✅"

                    print(f"   {safety_status} {name}")
                    print(f"      경로: {config.model_path}")
                    print(f"      유형: {config.model_type}")
                    print(f"      장치: {config.device}")
                    print(f"      양자화: 4bit={config.load_in_4bit}, 8bit={config.load_in_8bit}")
                    if warnings:
                        print(f"      경고: {len(warnings)}개")
                    print()

            elif args.type == 'datasets':
                datasets = self.config_manager.test_configs
                print(f"📋 등록된 데이터셋 ({len(datasets)}개):")

                for name, config in datasets.items():
                    data_file = Path(config.dataset_path)
                    exists = "✅" if data_file.exists() else "❌"
                    print(f"   {exists} {name}")
                    print(f"      경로: {config.dataset_path}")
                    print(f"      샘플: {config.num_samples}개")
                    print()

            elif args.type == 'results':
                results_dir = Path("optimization_results")
                if results_dir.exists():
                    result_files = list(results_dir.glob("*.json"))
                    print(f"📋 저장된 결과 ({len(result_files)}개):")

                    for result_file in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                        try:
                            with open(result_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)

                            test_type = "🔧" if 'best_score' in data else "⚡"
                            model = data.get('model_name', 'Unknown')
                            dataset = data.get('dataset_name', 'Unknown')
                            timestamp = data.get('timestamp', '')[:16].replace('T', ' ')

                            print(f"   {test_type} {result_file.name}")
                            print(f"      {model} on {dataset} ({timestamp})")

                        except Exception:
                            print(f"   ❌ {result_file.name} (읽기 실패)")
                else:
                    print("📋 저장된 결과가 없습니다.")

        except Exception as e:
            self.logger.error(f"정보 조회 실패: {e}")
            print(f"❌ 정보 조회 실패: {e}")


    def run_status_command(self, args):
        """status 명령어 실행"""
        print("🔍 시스템 상태 점검")
        self.show_system_status()

        # 추가 상태 정보
        try:
            from config import get_resource_manager
            resource_manager = get_resource_manager()
            memory_usage = resource_manager.get_memory_usage()

            if memory_usage:
                print(f"\n🎯 현재 리소스 사용량:")
                for key, value in memory_usage.items():
                    if isinstance(value, float) and 'percent' in key:
                        status = "⚠️" if value > 80 else "✅"
                        print(f"   {status} {key}: {value:.1f}%")
                    elif isinstance(value, float) and 'gb' in key:
                        print(f"      {key}: {value:.2f}GB")

        except Exception as e:
            print(f"⚠️ 리소스 정보 확인 실패: {e}")


    def run_clean_command(self, args):
        """clean 명령어 실행"""
        print("🧹 시스템 정리 시작...")

        cleaned_items = []

        try:
            if args.cache or args.all:
                # CUDA 캐시 정리
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        cleaned_items.append("CUDA 캐시")
                except:
                    pass

                # Python 가비지 컬렉션
                gc.collect()
                cleaned_items.append("Python 가비지")

            if args.logs or args.all:
                # 오래된 로그 파일 정리 (7일 이상)
                log_dir = Path("logs")
                if log_dir.exists():
                    import time
                    current_time = time.time()
                    old_logs = []

                    for log_file in log_dir.glob("*.log"):
                        if current_time - log_file.stat().st_mtime > 7 * 24 * 3600:  # 7일
                            old_logs.append(log_file)

                    for log_file in old_logs:
                        log_file.unlink()

                    if old_logs:
                        cleaned_items.append(f"{len(old_logs)}개 오래된 로그 파일")

            if args.results or args.all:
                # 사용자 확인 후 결과 정리
                results_dir = Path("optimization_results")
                if results_dir.exists():
                    result_files = list(results_dir.glob("*.json"))
                    if result_files:
                        response = input(f"⚠️ {len(result_files)}개 결과 파일을 정리하시겠습니까? (y/N): ")
                        if response.lower() == 'y':
                            for result_file in result_files:
                                result_file.unlink()
                            cleaned_items.append(f"{len(result_files)}개 결과 파일")

            # HuggingFace 캐시 정리 (선택적)
            if args.all:
                hf_cache = Path.home() / ".cache" / "huggingface"
                if hf_cache.exists():
                    response = input("⚠️ HuggingFace 캐시를 정리하시겠습니까? (y/N): ")
                    if response.lower() == 'y':
                        import shutil
                        shutil.rmtree(hf_cache, ignore_errors=True)
                        cleaned_items.append("HuggingFace 캐시")

            if cleaned_items:
                print("✅ 정리 완료:")
                for item in cleaned_items:
                    print(f"   - {item}")
            else:
                print("✅ 정리할 항목이 없습니다.")

        except Exception as e:
            self.logger.error(f"정리 작업 실패: {e}")
            print(f"❌ 정리 작업 실패: {e}")


    def show_welcome_screen(self):
        """환영 화면"""
        self.print_banner()
        print("🛡️ 안전한 오픈소스 LLM 추론 성능 최적화 시스템에 오신 것을 환영합니다!")

        print("\n🚀 빠른 시작 (안전 모드):")
        print("1. 시스템 초기화: python safe_main.py init --auto-detect")
        print("2. 하드웨어 확인: python safe_main.py hardware")
        print("3. 시스템 상태: python safe_main.py status")
        print("4. 안전한 최적화: python safe_main.py optimize --model qwen2.5-7b --dataset korean_math --samples 10 --safe")
        print("5. 성능 벤치마크: python safe_main.py benchmark --model qwen2.5-7b --dataset korean_qa --samples 20")

        print("\n🔧 주요 개선사항:")
        print("   ✅ 메모리 누수 방지 - 자동 리소스 정리")
        print("   ✅ 스레드 안전성 - 동시성 문제 해결")
        print("   ✅ 의존성 안정성 - Optuna 기반 최적화")
        print("   ✅ 오류 복구 - 강화된 예외 처리")
        print("   ✅ 보안 강화 - 입력 검증 및 안전 모드")

        print("\n⚡ 안전 사용 팁:")
        print("   🛡️ 처음 사용: --safe 옵션 필수")
        print("   💾 메모리 절약: --samples 10-20 권장")
        print("   🐛 문제 해결: --debug 옵션 활용")
        print("   🔧 시스템 정리: clean 명령어 정기 실행")

        print("\n💡 도움말:")
        print("   전체 명령어: python safe_main.py --help")
        print("   명령어별 도움말: python safe_main.py [명령어] --help")

        # 시스템 상태 간단히 표시
        self.show_system_status()


    async def main(self):
        """메인 함수"""
        parser = self.create_argument_parser()

        # 인자가 없으면 환영 화면
        if len(sys.argv) == 1:
            self.show_welcome_screen()
            return

        try:
            args = parser.parse_args()
        except SystemExit:
            return

        # 로깅 설정
        setup_safe_logging(args.debug)
        self.logger = logging.getLogger(__name__)

        if not args.command:
            self.print_banner()
            parser.print_help()
            return

        self.print_banner()

        # 독립적 명령어들 (시스템 요구사항 확인 불필요)
        if args.command in ['init', 'hardware', 'list', 'status', 'clean']:
            try:
                if args.command == 'init':
                    self.run_init_command(args)
                elif args.command == 'hardware':
                    self.run_hardware_command(args)
                elif args.command == 'list':
                    self.run_list_command(args)
                elif args.command == 'status':
                    self.run_status_command(args)
                elif args.command == 'clean':
                    self.run_clean_command(args)
                return
            except KeyboardInterrupt:
                print("\n⏹️ 사용자에 의해 중단되었습니다.")
                return
            except Exception as e:
                self.logger.error(f"명령 실행 실패: {e}")
                print(f"❌ 명령 실행 실패: {e}")
                if args.debug:
                    traceback.print_exc()
                return

        # 복잡한 명령어들 (시스템 요구사항 확인 필요)
        if not self.check_system_requirements():
            print("\n⚠️ 시스템 요구사항을 만족하지 않습니다.")
            if not args.safe:
                response = input("안전 모드로 계속 진행하시겠습니까? (y/N): ")
                if response.lower() != 'y':
                    return
                args.safe = True

        try:
            # 타임아웃 설정
            timeout_task = None
            if args.timeout > 0:
                timeout_task = asyncio.create_task(asyncio.sleep(args.timeout))

            # 명령 실행
            if args.command == 'optimize':
                command_task = asyncio.create_task(self.run_optimize_command(args))
            elif args.command == 'benchmark':
                command_task = asyncio.create_task(self.run_benchmark_command(args))
            elif args.command == 'compare':
                command_task = asyncio.create_task(self.run_compare_command(args))
            else:
                print(f"❌ 알 수 없는 명령어: {args.command}")
                return

            # 타임아웃 또는 명령 완료 대기
            if timeout_task:
                done, pending = await asyncio.wait([command_task, timeout_task], return_when=asyncio.FIRST_COMPLETED)

                if timeout_task in done:
                    command_task.cancel()
                    print(f"\n⏰ 시간 초과 ({args.timeout}초)")
                else:
                    timeout_task.cancel()
            else:
                await command_task

        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.logger.error(f"실행 중 오류: {e}")
            print(f"❌ 실행 중 오류: {e}")
            if args.debug:
                traceback.print_exc()
        finally:
            # 정리 작업
            print("\n🧹 시스템 정리 중...")
            try:
                if self.optimizer:
                    # 최적화기 정리는 자동으로 처리됨
                    pass
                cleanup_all()
            except Exception as e:
                self.logger.error(f"정리 중 오류: {e}")
            print("✅ 정리 완료")


def main():
    """진입점"""
    try:
        cli = SafeCLI()
        asyncio.run(cli.main())
    except KeyboardInterrupt:
        print("\n⏹️ 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
        print("🛡️ 안전 모드로 다시 시도해보세요: --safe")
    finally:
        # 최종 정리
        cleanup_all()


if __name__ == "__main__":
    main()  # !/usr/bin/env python3
"""
안전성 강화된 오픈소스 LLM 추론 성능 최적화 시스템 메인 CLI
모든 주요 문제점이 해결된 안전한 버전
"""
import asyncio
import argparse
import sys
import json
import logging
import traceback
import gc
import os
import signal
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import threading
import atexit


# 안전한 로깅 설정
def setup_safe_logging(debug: bool = False):
    """안전한 로깅 설정"""
    level = logging.DEBUG if debug else logging.INFO

    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 로그 파일명 (날짜별)
    log_file = log_dir / f"llm_optimizer_{datetime.now().strftime('%Y%m%d')}.log"

    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 로깅 설정
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

    # 외부 라이브러리 로깅 레벨 조정
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('optuna').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


# 전역 정리 함수
_cleanup_functions = []
_cleanup_lock = threading.Lock()


def register_cleanup(func):
    """정리 함수 등록"""
    with _cleanup_lock:
        _cleanup_functions.append(func)


def cleanup_all():
    """모든 등록된 정리 함수 실행"""
    with _cleanup_lock:
        for func in _cleanup_functions:
            try:
                func()
            except Exception as e:
                print(f"Cleanup error: {e}")
        _cleanup_functions.clear()


# 시그널 핸들러
def signal_handler(signum, frame):
    """안전한 시그널 처리"""
    print(f"\n🛑 Signal {signum} received, cleaning up...")
    cleanup_all()
    sys.exit(0)


# 시그널 등록 (안전하게)
try:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
except ValueError:
    # 메인 스레드가 아닌 경우 무시
    pass
atexit.register(cleanup_all)


# 안전한 import with fallback
def safe_import():
    """안전한 모듈 import"""
    try:
        # 기본 모듈들 먼저 확인
        import torch
        import numpy as np
        import pandas as pd

        # 프로젝트 모듈들 import
        from safe_config import ConfigManager, HardwareDetector, cleanup_resources
        from safe_optimizer import SafeOptimizer

        register_cleanup(cleanup_resources)
        return True, None

    except ImportError as e:
        error_msg = f"Critical import error: {e}"
        missing_files = []

        required_files = [
            "safe_config.py",
            "safe_model_interface.py",
            "safe_optimizer.py",
            "dataset_loader.py"
        ]

        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)

        if missing_files:
            error_msg += f"\nMissing files: {', '.join(missing_files)}"

        return False, error_msg


# Import 시도
IMPORTS_OK, IMPORT_ERROR = safe_import()


class SafeCLI:
    """안전한 CLI 인터페이스"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_manager = None
        self.optimizer = None

    def create_argument_parser(self) -> argparse.ArgumentParser:
        """CLI 인자 파서 생성"""
        parser = argparse.ArgumentParser(
            description='안전한 오픈소스 LLM 추론 성능 최적화 시스템',
            formatter_class=argparse.RawDescriptionHelp