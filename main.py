#!/usr/bin/env python3
"""
오픈소스 LLM 추론 성능 최적화 시스템 메인 CLI
"""
import asyncio
import argparse
import sys
import json
import traceback
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 로컬 모듈 import 시도
try:
    from config import ConfigManager, InferenceParams, HardwareDetector
    from test_runner import PerformanceOptimizer
    from visualization import ResultVisualizer
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("현재 디렉토리에서 실행하고 있는지 확인하세요.")
    print("또는 python -m pip install -r requirements.txt 를 실행하세요.")
    sys.exit(1)

def create_argument_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서 생성"""
    parser = argparse.ArgumentParser(
        description='오픈소스 LLM 추론 성능 최적화 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py init --auto-detect              # 시스템 초기화
  python main.py hardware                        # 하드웨어 정보 확인
  python main.py optimize --model llama2-7b --dataset korean_math
  python main.py benchmark --model mistral-7b --dataset korean_qa
  python main.py compare --models llama2-7b mistral-7b --dataset korean_math
  python main.py visualize --type dashboard      # 결과 시각화
  python main.py export --format csv             # 결과 내보내기
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')

    # init 명령어
    init_parser = subparsers.add_parser('init', help='시스템 초기화')
    init_parser.add_argument('--force', action='store_true', help='기존 설정 덮어쓰기')
    init_parser.add_argument('--auto-detect', action='store_true', help='하드웨어 자동 감지')

    # hardware 명령어
    hardware_parser = subparsers.add_parser('hardware', help='하드웨어 정보 확인')
    hardware_parser.add_argument('--model-size', choices=['7b', '13b', '70b'], help='모델 크기별 추천')

    # optimize 명령어
    optimize_parser = subparsers.add_parser('optimize', help='파라미터 최적화')
    optimize_parser.add_argument('--model', required=True, help='모델 이름')
    optimize_parser.add_argument('--dataset', required=True, help='데이터셋 이름')
    optimize_parser.add_argument('--strategy', choices=['bayesian', 'grid_search', 'evolutionary'],
                                default='bayesian', help='최적화 전략')
    optimize_parser.add_argument('--trials', type=int, default=20, help='최적화 시도 횟수')
    optimize_parser.add_argument('--samples', type=int, default=50, help='테스트 샘플 수')
    optimize_parser.add_argument('--evaluator', default='korean_math', help='평가자 유형')

    # benchmark 명령어
    benchmark_parser = subparsers.add_parser('benchmark', help='성능 벤치마크')
    benchmark_parser.add_argument('--model', required=True, help='모델 이름')
    benchmark_parser.add_argument('--dataset', required=True, help='데이터셋 이름')
    benchmark_parser.add_argument('--samples', type=int, default=100, help='테스트 샘플 수')
    benchmark_parser.add_argument('--iterations', type=int, default=3, help='반복 횟수')
    benchmark_parser.add_argument('--temperature', type=float, default=0.1, help='Temperature')
    benchmark_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p')
    benchmark_parser.add_argument('--max-tokens', type=int, default=512, help='최대 토큰 수')

    # compare 명령어
    compare_parser = subparsers.add_parser('compare', help='모델 비교')
    compare_parser.add_argument('--models', nargs='+', required=True, help='비교할 모델들')
    compare_parser.add_argument('--dataset', required=True, help='데이터셋 이름')
    compare_parser.add_argument('--samples', type=int, default=100, help='테스트 샘플 수')
    compare_parser.add_argument('--metric', choices=['accuracy', 'speed', 'efficiency'],
                               default='accuracy', help='비교 기준')

    # list 명령어
    list_parser = subparsers.add_parser('list', help='정보 조회')
    list_parser.add_argument('--type', choices=['models', 'datasets', 'results'],
                           default='models', help='조회할 정보 유형')

    # visualize 명령어
    visualize_parser = subparsers.add_parser('visualize', help='결과 시각화')
    visualize_parser.add_argument('--type', choices=['optimization', 'benchmark', 'comparison', 'dashboard'],
                                 default='dashboard', help='시각화 유형')
    visualize_parser.add_argument('--output', help='출력 파일명')

    # export 명령어
    export_parser = subparsers.add_parser('export', help='결과 내보내기')
    export_parser.add_argument('--format', choices=['csv', 'json', 'excel'], 
                              default='csv', help='내보내기 형식')
    export_parser.add_argument('--output', help='출력 파일 경로')

    # analyze 명령어
    analyze_parser = subparsers.add_parser('analyze', help='결과 분석')
    analyze_parser.add_argument('--model', help='특정 모델 분석')
    analyze_parser.add_argument('--dataset', help='특정 데이터셋 분석')
    analyze_parser.add_argument('--report', action='store_true', help='상세 리포트 생성')

    return parser

def print_banner():
    """시스템 배너 출력"""
    banner = """
╭─────────────────────────────────────────────────────────────╮
│              🚀 오픈소스 LLM 추론 성능 최적화 시스템           │
│                 Open Source LLM Optimization                │
╰─────────────────────────────────────────────────────────────╯
"""
    print(banner)

def check_system_requirements() -> bool:
    """시스템 요구사항 확인"""
    requirements_met = True

    # Python 버전 확인
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8 이상이 필요합니다. 현재: {sys.version}")
        requirements_met = False

    # 필수 패키지 확인
    required_packages = ['torch', 'transformers', 'numpy', 'pandas']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ 누락된 패키지: {', '.join(missing_packages)}")
        print(f"설치 명령: pip install {' '.join(missing_packages)}")
        requirements_met = False

    return requirements_met

def show_system_status():
    """시스템 상태 표시"""
    print("🔧 시스템 상태 확인:")

    # 설정 파일 확인
    config_file = Path("llm_config.json")
    if config_file.exists():
        print("   ✅ 설정 파일 존재")
    else:
        print("   ❌ 설정 파일 없음 (python main.py init 실행 필요)")

    # 디렉토리 확인
    required_dirs = ["data", "optimization_results", "visualizations"]
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"   ✅ {dir_name} 디렉토리 존재")
        else:
            print(f"   ❌ {dir_name} 디렉토리 없음")

    # 하드웨어 정보
    try:
        hardware_info = HardwareDetector.detect_hardware()
        print(f"   💻 GPU: {hardware_info['cuda_device_count']}개")
        print(f"   💾 메모리: {hardware_info['available_memory']}/{hardware_info['total_memory']}GB")
    except Exception:
        print("   ⚠️  하드웨어 정보 확인 불가")

    # 결과 파일 확인
    results_dir = Path("optimization_results")
    if results_dir.exists():
        result_count = len(list(results_dir.glob("*.json")))
        print(f"   📊 저장된 결과: {result_count}개")
    else:
        print("   📊 저장된 결과: 0개")

def run_init_command(args):
    """init 명령어 실행"""
    print("🔧 시스템 초기화 시작...")

    config_file = "llm_config.json"

    if Path(config_file).exists() and not args.force:
        print(f"⚠️  설정 파일이 이미 존재합니다: {config_file}")
        print("   덮어쓰려면 --force 옵션을 사용하세요.")
        return

    try:
        # 설정 매니저 생성
        config_manager = ConfigManager(config_file)
        print(f"✅ 설정 파일 생성: {config_file}")

        # 디렉토리 생성
        directories = ["data", "optimization_results", "visualizations"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"   ✅ {directory} 디렉토리 생성")

        # 하드웨어 자동 감지
        if args.auto_detect:
            print("🔍 하드웨어 자동 감지 중...")
            try:
                hardware_info = HardwareDetector.detect_hardware()
                print(f"   GPU: {hardware_info['cuda_device_count']}개")
                print(f"   메모리: {hardware_info['total_memory']}GB")
                if hardware_info['cuda_available']:
                    total_gpu_memory = sum(
                        hardware_info.get(f'gpu_{i}_memory', 0) 
                        for i in range(hardware_info['cuda_device_count'])
                    )
                    print(f"   GPU 메모리: {total_gpu_memory}GB")
            except Exception as e:
                print(f"⚠️  하드웨어 감지 실패: {e}")

        print("\n📝 다음 단계:")
        print("1. 하드웨어 정보 확인: python main.py hardware")
        print("2. 모델 목록 확인: python main.py list --type models")
        print("3. 최적화 실행: python main.py optimize --model [모델명] --dataset [데이터셋명]")

    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        traceback.print_exc()
        sys.exit(1)

def run_hardware_command(args):
    """hardware 명령어 실행"""
    print("💻 하드웨어 정보 분석")

    try:
        hardware_info = HardwareDetector.detect_hardware()

        print(f"\n🔍 감지된 하드웨어:")
        print(f"   CUDA 사용 가능: {'✅' if hardware_info['cuda_available'] else '❌'}")
        print(f"   GPU 개수: {hardware_info['cuda_device_count']}")
        print(f"   총 메모리: {hardware_info['total_memory']}GB")
        print(f"   사용 가능 메모리: {hardware_info['available_memory']}GB")
        print(f"   CPU 코어: {hardware_info['cpu_cores']}")

        if hardware_info['cuda_available']:
            total_gpu_memory = 0
            for i in range(hardware_info['cuda_device_count']):
                gpu_memory = hardware_info.get(f'gpu_{i}_memory', 0)
                gpu_name = hardware_info.get(f'gpu_{i}_name', 'Unknown')
                print(f"   GPU {i}: {gpu_name} ({gpu_memory}GB)")
                total_gpu_memory += gpu_memory

            print(f"\n🎯 모델 추천:")
            if total_gpu_memory >= 80:
                print("   ✅ 70B 모델까지 실행 가능")
            elif total_gpu_memory >= 32:
                print("   ✅ 13B 모델까지 실행 가능")
            elif total_gpu_memory >= 16:
                print("   ✅ 7B 모델 실행 가능 (float16)")
            elif total_gpu_memory >= 8:
                print("   ⚠️  7B 모델 실행 가능 (4-bit 양자화 필요)")
            else:
                print("   ❌ GPU 메모리 부족, CPU 추론 권장")

        # 모델 크기별 추천 설정
        if hasattr(args, 'model_size') and args.model_size:
            print(f"\n🎯 {args.model_size.upper()} 모델 추천 설정:")
            recommended = HardwareDetector.recommend_config(args.model_size)
            print(f"   장치: {recommended.device}")
            print(f"   데이터 타입: {recommended.dtype}")
            print(f"   4-bit 양자화: {'✅' if recommended.load_in_4bit else '❌'}")
            print(f"   8-bit 양자화: {'✅' if recommended.load_in_8bit else '❌'}")

    except Exception as e:
        print(f"❌ 하드웨어 정보 확인 실패: {e}")

async def run_optimize_command(args):
    """optimize 명령어 실행"""
    print(f"🔧 파라미터 최적화 시작: {args.model} on {args.dataset}")
    print(f"   전략: {args.strategy}, 시도: {args.trials}회, 샘플: {args.samples}개")

    try:
        optimizer = PerformanceOptimizer()
        optimizer.setup_models()

        result = await optimizer.optimize_inference_params(
            model_name=args.model,
            dataset_name=args.dataset,
            evaluator_type=args.evaluator,
            optimization_strategy=args.strategy,
            max_trials=args.trials,
            num_samples=args.samples
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

        if result.recommendations:
            print(f"\n💡 추천사항:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"   {i}. {rec}")

        print(f"\n📁 결과 저장됨: optimization_results/{result.test_id}.json")

    except Exception as e:
        print(f"❌ 최적화 실패: {e}")
        traceback.print_exc()

async def run_benchmark_command(args):
    """benchmark 명령어 실행"""
    print(f"⚡ 벤치마크 시작: {args.model} on {args.dataset}")

    try:
        optimizer = PerformanceOptimizer()
        optimizer.setup_models()

        params = InferenceParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens
        )

        result = await optimizer.benchmark_model(
            model_name=args.model,
            dataset_name=args.dataset,
            params=params,
            num_samples=args.samples,
            iterations=args.iterations
        )

        print(f"✅ 벤치마크 완료!")

        perf = result.performance_metrics
        print(f"\n📊 성능 메트릭:")
        print(f"   토큰/초: {perf.tokens_per_second:.1f}")
        print(f"   지연시간 P95: {perf.latency_p95:.3f}초")
        print(f"   메모리 사용량: {perf.memory_usage_mb:.0f}MB")
        print(f"   처리량: {perf.throughput:.1f} req/sec")

        if result.cost_analysis:
            cost = result.cost_analysis
            print(f"\n💰 비용 분석:")
            print(f"   시간당 비용: ${cost.get('cost_per_hour_usd', 0):.4f}")
            print(f"   1K토큰당 비용: ${cost.get('cost_per_1k_tokens_usd', 0):.6f}")

        accuracy = sum(1 for r in result.evaluation_results if r.correct) / len(result.evaluation_results)
        print(f"\n🎯 정확도: {accuracy:.3f}")

        print(f"\n📁 결과 저장됨: optimization_results/bench_{result.test_id}.json")

    except Exception as e:
        print(f"❌ 벤치마크 실패: {e}")
        traceback.print_exc()

async def run_compare_command(args):
    """compare 명령어 실행"""
    print(f"⚖️ 모델 비교 시작: {', '.join(args.models)} on {args.dataset}")

    try:
        optimizer = PerformanceOptimizer()
        optimizer.setup_models()

        results = {}
        for model in args.models:
            print(f"\n🔄 {model} 테스트 중...")

            params = InferenceParams(temperature=0.1, top_p=0.9, max_new_tokens=512)

            result = await optimizer.benchmark_model(
                model_name=model,
                dataset_name=args.dataset,
                params=params,
                num_samples=args.samples,
                iterations=1
            )

            results[model] = result

            perf = result.performance_metrics
            accuracy = sum(1 for r in result.evaluation_results if r.correct) / len(result.evaluation_results)
            print(f"   ✅ 완료: 정확도 {accuracy:.3f}, {perf.tokens_per_second:.1f} tokens/sec")

        # 결과 정렬 및 출력
        print(f"\n📊 비교 결과 ({args.metric} 기준):")
        print(f"{'순위':<4} {'모델':<20} {'정확도':<8} {'토큰/초':<10} {'메모리(MB)':<12}")
        print("-" * 60)

        if args.metric == 'accuracy':
            sorted_results = sorted(results.items(), 
                                  key=lambda x: sum(1 for r in x[1].evaluation_results if r.correct) / len(x[1].evaluation_results),
                                  reverse=True)
        elif args.metric == 'speed':
            sorted_results = sorted(results.items(), 
                                  key=lambda x: x[1].performance_metrics.tokens_per_second,
                                  reverse=True)
        else:
            sorted_results = list(results.items())

        for i, (model, result) in enumerate(sorted_results, 1):
            perf = result.performance_metrics
            accuracy = sum(1 for r in result.evaluation_results if r.correct) / len(result.evaluation_results)

            rank_symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{rank_symbol:<4} {model:<20} {accuracy:<8.3f} {perf.tokens_per_second:<10.1f} {perf.memory_usage_mb:<12.0f}")

    except Exception as e:
        print(f"❌ 모델 비교 실패: {e}")
        traceback.print_exc()

def run_list_command(args):
    """list 명령어 실행"""
    try:
        config_manager = ConfigManager()

        if args.type == 'models':
            models = config_manager.model_configs
            print(f"📋 등록된 모델 ({len(models)}개):")
            for name, config in models.items():
                print(f"   📦 {name}")
                print(f"      경로: {config.model_path}")
                print(f"      유형: {config.model_type}")
                print(f"      장치: {config.device}")
                print(f"      타입: {config.dtype}")
                print()

        elif args.type == 'datasets':
            datasets = config_manager.test_configs
            print(f"📋 등록된 데이터셋 ({len(datasets)}개):")
            for name, config in datasets.items():
                print(f"   📄 {name}")
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
        print(f"❌ 정보 조회 실패: {e}")

def run_visualize_command(args):
    """visualize 명령어 실행"""
    print(f"📊 시각화 생성: {args.type}")

    try:
        visualizer = ResultVisualizer("optimization_results")
        output_name = args.output or f"{args.type}_output"

        if args.type == 'optimization':
            results = visualizer.load_optimization_results()
            if results:
                fig = visualizer.create_optimization_analysis(results, output_name)
                print(f"✅ 최적화 분석 차트 생성: visualizations/{output_name}.html")
            else:
                print("❌ 최적화 결과가 없습니다.")

        elif args.type == 'benchmark':
            results = visualizer.load_benchmark_results()
            if results:
                fig = visualizer.create_benchmark_analysis(results, output_name)
                print(f"✅ 벤치마크 분석 차트 생성: visualizations/{output_name}.html")
            else:
                print("❌ 벤치마크 결과가 없습니다.")

        elif args.type == 'comparison':
            results = visualizer.load_all_results()
            if results:
                fig = visualizer.create_model_comparison_chart(results, output_name)
                print(f"✅ 모델 비교 차트 생성: visualizations/{output_name}.html")
            else:
                print("❌ 비교할 결과가 없습니다.")

        elif args.type == 'dashboard':
            fig = visualizer.create_comprehensive_dashboard(output_name)
            print(f"✅ 종합 대시보드 생성: visualizations/{output_name}.html")

    except Exception as e:
        print(f"❌ 시각화 실패: {e}")
        traceback.print_exc()

def run_export_command(args):
    """export 명령어 실행"""
    print(f"📤 결과 내보내기: {args.format} 형식")

    try:
        results_dir = Path("optimization_results")
        if not results_dir.exists():
            print("❌ 내보낼 결과가 없습니다.")
            return

        all_results = []
        for result_file in results_dir.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                flat_data = {
                    'file_name': result_file.name,
                    'test_id': data.get('test_id', ''),
                    'model_name': data.get('model_name', ''),
                    'dataset_name': data.get('dataset_name', ''),
                    'timestamp': data.get('timestamp', ''),
                    'test_type': 'optimization' if 'best_score' in data else 'benchmark'
                }

                if 'best_score' in data:
                    flat_data.update({
                        'best_score': data.get('best_score', 0),
                        'total_time': data.get('total_time', 0),
                        'temperature': data.get('best_params', {}).get('temperature', 0),
                        'top_p': data.get('best_params', {}).get('top_p', 0),
                        'top_k': data.get('best_params', {}).get('top_k', 0)
                    })

                elif 'performance_metrics' in data:
                    perf = data.get('performance_metrics', {})
                    flat_data.update({
                        'tokens_per_second': perf.get('tokens_per_second', 0),
                        'latency_p95': perf.get('latency_p95', 0),
                        'memory_usage_mb': perf.get('memory_usage_mb', 0)
                    })

                all_results.append(flat_data)

            except Exception as e:
                print(f"⚠️  파일 {result_file.name} 처리 실패: {e}")

        if not all_results:
            print("❌ 내보낼 데이터가 없습니다.")
            return

        # 출력 파일명 결정
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"llm_results_{timestamp}.{args.format}")

        # 형식별 내보내기
        if args.format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        elif args.format == 'csv':
            try:
                import pandas as pd
                df = pd.DataFrame(all_results)
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
            except ImportError:
                print("❌ pandas가 설치되지 않았습니다. pip install pandas")
                return

        elif args.format == 'excel':
            try:
                import pandas as pd
                df = pd.DataFrame(all_results)
                df.to_excel(output_path, index=False)
            except ImportError:
                print("❌ pandas와 openpyxl이 설치되지 않았습니다. pip install pandas openpyxl")
                return

        print(f"✅ 결과 내보내기 완료: {output_path}")
        print(f"   총 {len(all_results)}개 결과 내보냄")

    except Exception as e:
        print(f"❌ 내보내기 실패: {e}")
        traceback.print_exc()

def run_analyze_command(args):
    """analyze 명령어 실행"""
    print("🔍 결과 분석 시작...")

    try:
        results_dir = Path("optimization_results")
        if not results_dir.exists():
            print("❌ 분석할 결과가 없습니다.")
            return

        optimization_results = []
        benchmark_results = []

        for result_file in results_dir.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 필터 적용
                if args.model and args.model.lower() not in data.get('model_name', '').lower():
                    continue
                if args.dataset and args.dataset.lower() not in data.get('dataset_name', '').lower():
                    continue

                if 'best_score' in data:
                    optimization_results.append(data)
                elif 'performance_metrics' in data:
                    benchmark_results.append(data)

            except Exception:
                continue

        print(f"📊 분석 대상: 최적화 {len(optimization_results)}개, 벤치마크 {len(benchmark_results)}개")

        # 최적화 결과 분석
        if optimization_results:
            print(f"\n🔧 최적화 결과 분석:")
            best_result = max(optimization_results, key=lambda x: x.get('best_score', 0))
            avg_score = sum(r.get('best_score', 0) for r in optimization_results) / len(optimization_results)

            print(f"   최고 점수: {best_result.get('best_score', 0):.3f} ({best_result.get('model_name', 'Unknown')})")
            print(f"   평균 점수: {avg_score:.3f}")

            best_params = best_result.get('best_params', {})
            print(f"   최적 파라미터:")
            print(f"     Temperature: {best_params.get('temperature', 0):.3f}")
            print(f"     Top-p: {best_params.get('top_p', 0):.3f}")
            print(f"     Top-k: {best_params.get('top_k', 0)}")

        # 벤치마크 결과 분석
        if benchmark_results:
            print(f"\n⚡ 벤치마크 결과 분석:")
            fastest_model = max(benchmark_results, 
                              key=lambda x: x.get('performance_metrics', {}).get('tokens_per_second', 0))

            fastest_speed = fastest_model.get('performance_metrics', {}).get('tokens_per_second', 0)
            print(f"   최고 속도: {fastest_speed:.1f} tokens/sec ({fastest_model.get('model_name', 'Unknown')})")

        print(f"\n💡 추천사항:")
        print("   1. 정확도와 속도의 균형을 고려한 파라미터 사용")
        print("   2. 정기적인 성능 모니터링")
        print("   3. 하드웨어 사용량 최적화")

        # 상세 리포트 생성
        if args.report:
            try:
                visualizer = ResultVisualizer("optimization_results")
                report_path = visualizer.generate_performance_report("analysis_report.html")
                print(f"\n📄 상세 리포트 생성: {report_path}")
            except Exception as e:
                print(f"⚠️  리포트 생성 실패: {e}")

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        traceback.print_exc()

def show_welcome_screen():
    """환영 화면 표시"""
    print_banner()
    print("🚀 오픈소스 LLM 추론 성능 최적화 시스템에 오신 것을 환영합니다!")

    print("\n📖 빠른 시작 가이드:")
    print("1. 시스템 초기화: python main.py init --auto-detect")
    print("2. 하드웨어 확인: python main.py hardware")
    print("3. 모델 목록 확인: python main.py list --type models")
    print("4. 최적화 실행: python main.py optimize --model [모델명] --dataset [데이터셋명]")
    print("5. 결과 확인: python main.py visualize --type dashboard")

    print("\n💡 도움말:")
    print("   전체 명령어: python main.py --help")
    print("   특정 명령어: python main.py [명령어] --help")

    # 시스템 상태 확인
    show_system_status()

    print("\n🎯 주요 기능:")
    print("   🔧 파라미터 최적화 - 베이지안/그리드서치/진화 알고리즘")
    print("   ⚡ 성능 벤치마크 - 속도/메모리/비용 분석")
    print("   ⚖️ 모델 비교 - 다중 모델 성능 비교")
    print("   📊 결과 시각화 - 인터랙티브 차트 및 대시보드")
    print("   📤 결과 내보내기 - CSV/JSON/Excel 형식 지원")

async def main():
    """메인 함수"""
    # Python 버전 확인
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        print(f"현재 버전: {sys.version}")
        sys.exit(1)

    parser = create_argument_parser()

    # 인자가 없으면 환영 화면 표시
    if len(sys.argv) == 1:
        show_welcome_screen()
        return

    try:
        args = parser.parse_args()
    except SystemExit:
        return

    if not args.command:
        print_banner()
        parser.print_help()
        return

    print_banner()

    # 독립적으로 실행 가능한 명령어들
    if args.command == 'init':
        run_init_command(args)
        return

    if args.command == 'hardware':
        run_hardware_command(args)
        return

    if args.command == 'list':
        run_list_command(args)
        return

    if args.command == 'visualize':
        run_visualize_command(args)
        return

    if args.command == 'export':
        run_export_command(args)
        return

    if args.command == 'analyze':
        run_analyze_command(args)
        return

    # 시스템 요구사항 확인
    if not check_system_requirements():
        print("⚠️  시스템 요구사항을 만족하지 않습니다. 계속 진행하시겠습니까? (y/N)")
        if input().lower() != 'y':
            sys.exit(1)

    # PerformanceOptimizer가 필요한 명령어들
    try:
        print("🔧 시스템 초기화 중...")

        # PerformanceOptimizer 실행 전 설정 파일 확인
        config_file = Path("llm_config.json")
        if not config_file.exists():
            print("❌ 설정 파일이 없습니다.")
            print("   python main.py init 명령어를 먼저 실행하세요.")
            return

        # PerformanceOptimizer 초기화는 각 명령어에서 수행
        if args.command == 'optimize':
            await run_optimize_command(args)
        elif args.command == 'benchmark':
            await run_benchmark_command(args)
        elif args.command == 'compare':
            await run_compare_command(args)
        else:
            print(f"❌ 알 수 없는 명령어: {args.command}")

    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        if "--debug" in sys.argv:
            traceback.print_exc()
        else:
            print("상세 오류 정보: python main.py [명령어] --debug")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 프로그램이 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
        if "--debug" in sys.argv:
            traceback.print_exc()
        sys.exit(1)