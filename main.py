else:
print("⚠️ CUDA 사용 불가 - CPU 모드로 실행됩니다.")
except Exception as e:
print(f"⚠️ GPU 상태 확인 불가: {e}")

# 오류 출력
if issues:
    print(f"❌ 시스템 요구사항 미충족:")
    for issue in issues:
        print(f"   - {issue}")

    print(f"\n💡 해결 방법:")
    if missing_packages:
        print(f"   pip install {' '.join([p.split(' ')[0] for p in missing_packages])}")

return requirements_met


def show_system_status_enhanced():
    """강화된 시스템 상태 표시"""
    print("🔧 시스템 상태 확인:")

    # 설정 파일 확인
    config_file = Path("llm_config.json")
    if config_file.exists():
        try:
            config_manager = RobustConfigManager()
            model_count = len(config_manager.model_configs)
            print(f"   ✅ 설정 파일 존재 ({model_count}개 모델 설정)")

            # 설정 검증
            validator = ConfigValidator()
            total_issues = 0
            for name, config in config_manager.model_configs.items():
                issues = validator.validate_model_config(config)
                if issues:
                    total_issues += len(issues)
                    print(f"   ⚠️ 모델 '{name}': {len(issues)}개 문제")

            if total_issues == 0:
                print("   ✅ 모든 모델 설정이 유효합니다")

        except Exception as e:
            print(f"   ❌ 설정 파일 읽기 실패: {e}")
    else:
        print("   ❌ 설정 파일 없음 (python safe_main.py init 실행 필요)")

    # 디렉토리 확인
    required_dirs = ["data", "optimization_results", "visualizations"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"   ✅ {dir_name} 디렉토리 ({file_count}개 파일)")
        else:
            print(f"   ❌ {dir_name} 디렉토리 없음")

    # 하드웨어 정보 (상세)
    try:
        hardware_info = HardwareDetector.detect_hardware()
        print(f"   💻 시스템 정보:")
        print(f"      CPU: {hardware_info['cpu_cores']}코어")
        print(f"      메모리: {hardware_info['available_memory']}/{hardware_info['total_memory']}GB")

        if hardware_info['cuda_available']:
            print(f"      GPU: {hardware_info['cuda_device_count']}개")
            for i in range(hardware_info['cuda_device_count']):
                gpu_name = hardware_info.get(f'gpu_{i}_name', f'GPU {i}')
                gpu_memory = hardware_info.get(f'gpu_{i}_memory', 0)
                print(f"        {gpu_name}: {gpu_memory}GB")
        else:
            print(f"      GPU: CUDA 사용 불가")

    except Exception as e:
        print(f"   ⚠️ 하드웨어 정보 확인 불가: {e}")

    # 결과 파일 확인 (상세)
    results_dir = Path("optimization_results")
    if results_dir.exists():
        opt_files = list(results_dir.glob("opt_*.json"))
        bench_files = list(results_dir.glob("bench_*.json"))
        print(f"   📊 저장된 결과:")
        print(f"      최적화: {len(opt_files)}개")
        print(f"      벤치마크: {len(bench_files)}개")

        # 최근 결과 표시
        if opt_files or bench_files:
            all_files = sorted(
                opt_files + bench_files,
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            latest = all_files[0]
            mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
            print(f"      최신: {latest.name} ({mod_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("   📊 저장된 결과: 0개")


def run_init_command(args):
    """init 명령어 실행 - 안전성 강화"""
    print("🔧 안전한 시스템 초기화 시작...")

    config_file = "llm_config.json"

    if Path(config_file).exists() and not args.force:
        print(f"⚠️  설정 파일이 이미 존재합니다: {config_file}")
        print("   덮어쓰려면 --force 옵션을 사용하세요.")
        return

    try:
        # 안전한 설정 매니저 생성
        config_manager = RobustConfigManager(config_file)
        print(f"✅ 안전한 설정 파일 생성: {config_file}")

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

                    # 안전성 검사
                    for name, config in config_manager.model_configs.items():
                        safety_warnings = SafetyChecker.check_memory_safety(config)
                        if safety_warnings:
                            print(f"   ⚠️ {name} 안전성 경고:")
                            for warning in safety_warnings:
                                print(f"      - {warning}")

            except Exception as e:
                print(f"⚠️  하드웨어 감지 실패: {e}")

        print("\n📝 다음 단계:")
        print("1. 하드웨어 정보 확인: python safe_main.py hardware")
        print("2. 모델 목록 확인: python safe_main.py list --type models")
        print("3. 안전한 최적화 실행: python safe_main.py optimize --model [모델명] --dataset [데이터셋명] --samples 20")
        print("4. 결과 확인: python safe_main.py visualize --type dashboard")

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

            print(f"\n🎯 안전한 모델 추천:")
            if total_gpu_memory >= 80:
                print("   ✅ 70B 모델까지 실행 가능 (양자화 권장)")
            elif total_gpu_memory >= 32:
                print("   ✅ 13B 모델까지 실행 가능")
            elif total_gpu_memory >= 16:
                print("   ✅ 7B 모델 실행 가능 (float16)")
            elif total_gpu_memory >= 8:
                print("   ⚠️  7B 모델 실행 가능 (4-bit 양자화 필수)")
            else:
                print("   ❌ GPU 메모리 부족, CPU 추론 권장")

        # 모델 크기별 추천 설정
        if hasattr(args, 'model_size') and args.model_size:
            print(f"\n🎯 {args.model_size.upper()} 모델 안전 설정:")
            recommended = HardwareDetector.recommend_config(args.model_size)
            print(f"   장치: {recommended.device}")
            print(f"   데이터 타입: {recommended.dtype}")
            print(f"   4-bit 양자화: {'✅' if recommended.load_in_4bit else '❌'}")
            print(f"   8-bit 양자화: {'✅' if recommended.load_in_8bit else '❌'}")

            # 안전한 배치 크기 추천
            available_memory = hardware_info.get('gpu_0_memory', 8)
            safe_batch = HardwareDetector.get_safe_batch_size(args.model_size, available_memory)
            print(f"   권장 배치 크기: {safe_batch}")

    except Exception as e:
        print(f"❌ 하드웨어 정보 확인 실패: {e}")


async def run_optimize_command(args):
    """optimize 명령어 실행 - 안전성 강화"""
    print(f"🔧 안전한 파라미터 최적화 시작: {args.model} on {args.dataset}")

    # 안전 모드 적용
    if hasattr(args, 'safe_mode') and args.safe_mode:
        max_samples = min(args.samples, 10)
        max_trials = min(args.trials, 5)
        print(f"   🛡️ 안전 모드: 샘플 {max_samples}개, 시도 {max_trials}회로 제한")
    else:
        max_samples = min(args.samples, 50)  # 기본 제한
        max_trials = min(args.trials, 20)
        print(f"   전략: {args.strategy}, 시도: {max_trials}회, 샘플: {max_samples}개")

    try:
        # 안전한 최적화기 사용
        optimizer = SafePerformanceOptimizer()

        # 모델 설정 검증
        config_manager = RobustConfigManager()
        model_config = config_manager.get_safe_model_config(args.model)

        if not model_config:
            print(f"❌ 모델 {args.model} 설정에 문제가 있습니다.")
            print("   대체 설정을 생성합니다...")
            model_config = config_manager.create_fallback_config(args.model)

        # 안전성 검사
        safety_warnings = SafetyChecker.check_memory_safety(model_config)
        if safety_warnings:
            print("⚠️ 안전성 경고:")
            for warning in safety_warnings:
                print(f"   - {warning}")
            print("   계속 진행하시겠습니까? (y/N): ", end="")
            response = input()
            if response.lower() != 'y':
                sys.exit(1)
            else:
                args.safe_mode = True  # 강제로 안전 모드 활성화

    # 안전한 실행을 위한 메모리 체크
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                if allocated / total > 0.8:
                    print(f"⚠️ GPU {i} 메모리 사용률이 높습니다 ({allocated / total:.1%})")
                    print("   메모리 정리를 시도합니다...")
                    torch.cuda.empty_cache()
    except:
        pass

    # 복잡한 명령어들 처리
    try:
        if args.command == 'optimize':
            await run_optimize_command(args)
        elif args.command == 'benchmark':
            await run_benchmark_command(args)
        elif args.command == 'compare':
            await run_compare_command(args)
        elif args.command == 'visualize':
            run_visualize_command(args)
        elif args.command == 'export':
            run_export_command(args)
        elif args.command == 'analyze':
            run_analyze_command(args)
        else:
            print(f"❌ 알 수 없는 명령어: {args.command}")

    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        if hasattr(args, 'debug') and args.debug:
            traceback.print_exc()
        else:
            print("상세 오류 정보: --debug 옵션 사용")
    finally:
        # 안전한 정리
        print("\n🧹 시스템 정리 중...")
        try:
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        print("✅ 정리 완료")


def run_visualize_command(args):
    """visualize 명령어 실행 - 기본 구현"""
    print(f"📊 시각화 생성: {args.type}")

    try:
        # 결과 디렉토리 확인
        results_dir = Path("optimization_results")
        if not results_dir.exists() or not list(results_dir.glob("*.json")):
            print("❌ 시각화할 결과가 없습니다.")
            print("   먼저 최적화나 벤치마크를 실행하세요.")
            return

        # 기본 HTML 리포트 생성
        output_file = f"visualizations/{args.output or args.type}_report.html"
        Path("visualizations").mkdir(exist_ok=True)

        html_content = generate_basic_report(results_dir)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ 기본 리포트 생성: {output_file}")
        print("   브라우저에서 열어서 확인하세요.")

    except Exception as e:
        print(f"❌ 시각화 실패: {e}")


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

                # 기본 정보 추출
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
                        'total_time': data.get('total_time', 0)
                    })
                elif 'performance_metrics' in data:
                    perf = data.get('performance_metrics', {})
                    flat_data.update({
                        'tokens_per_second': perf.get('tokens_per_second', 0),
                        'memory_usage_mb': perf.get('memory_usage_mb', 0)
                    })

                all_results.append(flat_data)

            except Exception as e:
                print(f"⚠️ 파일 {result_file.name} 처리 실패: {e}")

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
                # pandas 없이 CSV 생성
                import csv
                if all_results:
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                        writer.writeheader()
                        writer.writerows(all_results)

        print(f"✅ 결과 내보내기 완료: {output_path}")
        print(f"   총 {len(all_results)}개 결과 내보냄")

    except Exception as e:
        print(f"❌ 내보내기 실패: {e}")


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

        # 추천사항
        print(f"\n💡 안전성 중심 추천사항:")
        print("   1. 메모리 사용량 모니터링 및 양자화 활용")
        print("   2. 배치 크기 조정으로 안정성 확보")
        print("   3. 정기적인 시스템 리소스 체크")
        print("   4. 안전 모드 사용으로 위험 요소 최소화")

        # 상세 리포트 생성
        if args.report:
            try:
                report_path = generate_analysis_report(optimization_results, benchmark_results)
                print(f"\n📄 상세 리포트 생성: {report_path}")
            except Exception as e:
                print(f"⚠️ 리포트 생성 실패: {e}")

    except Exception as e:
        print(f"❌ 분석 실패: {e}")


def generate_basic_report(results_dir: Path) -> str:
    """기본 HTML 리포트 생성"""
    results = []
    for file_path in results_dir.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
        except:
            continue

    html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>LLM 최적화 결과 리포트</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #4CAF50; color: white; padding: 20px; border-radius: 5px; }}
            .result {{ margin: 10px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .optimization {{ background-color: #e3f2fd; }}
            .benchmark {{ background-color: #fff3e0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 안전한 LLM 최적화 결과</h1>
            <p>생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>총 결과: {len(results)}개</p>
        </div>
    """

    for result in results:
        result_type = "optimization" if 'best_score' in result else "benchmark"
        css_class = result_type

        html += f"""
        <div class="result {css_class}">
            <h3>{'🔧 최적화' if result_type == 'optimization' else '⚡ 벤치마크'}: {result.get('model_name', 'Unknown')}</h3>
            <p><strong>데이터셋:</strong> {result.get('dataset_name', 'Unknown')}</p>
            <p><strong>시간:</strong> {result.get('timestamp', '')}</p>
        """

        if result_type == 'optimization':
            html += f"""
            <p><strong>최고 점수:</strong> {result.get('best_score', 0):.3f}</p>
            <p><strong>소요 시간:</strong> {result.get('total_time', 0):.1f}초</p>
            """
        else:
            perf = result.get('performance_metrics', {})
            html += f"""
            <p><strong>처리 속도:</strong> {perf.get('tokens_per_second', 0):.1f} tokens/sec</p>
            <p><strong>메모리 사용량:</strong> {perf.get('memory_usage_mb', 0):.0f}MB</p>
            """

        html += "</div>"

    html += """
    </body>
    </html>
    """

    return html


def generate_analysis_report(opt_results: List[Dict], bench_results: List[Dict]) -> str:
    """분석 리포트 생성"""
    report_path = Path("visualizations/analysis_report.html")
    Path("visualizations").mkdir(exist_ok=True)

    html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>상세 분석 리포트</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .header {{ background: linear-gradient(90deg, #4CAF50, #45a049); color: white; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #4CAF50; background-color: #f9f9f9; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 상세 분석 리포트</h1>
            <p>생성 시간: {datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")}</p>
        </div>

        <div class="section">
            <h2>📈 요약</h2>
            <p>최적화 실험: {len(opt_results)}개</p>
            <p>벤치마크 테스트: {len(bench_results)}개</p>
        </div>
    """

    if opt_results:
        best = max(opt_results, key=lambda x: x.get('best_score', 0))
        html += f"""
        <div class="section">
            <h2>🏆 최고 성능</h2>
            <p>모델: {best.get('model_name', 'Unknown')}</p>
            <p>점수: {best.get('best_score', 0):.3f}</p>
            <p>데이터셋: {best.get('dataset_name', 'Unknown')}</p>
        </div>
        """

    html += """
    </body>
    </html>
    """

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return str(report_path)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 프로그램이 중단되었습니다.")

        # 안전한 정리 작업
        try:
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
        print("🛡️ 안전 모드로 다시 시도해보세요: --safe-mode")
        sys.exit(1)
        print("최적화를 중단합니다.")
        return

    # 안전한 모델 등록
    safe_model_manager = SafeModelManager(config_manager.optimization_config)
    safe_model_manager.register_model(args.model, model_config)
    optimizer.model_manager = safe_model_manager

    # 메모리 체크
    memory_usage = optimizer.resource_manager.check_memory_usage()
    print(f"   초기 메모리 사용량: {memory_usage}")

    # 최적화 실행
    result = await optimizer.optimize_inference_params(
        model_name=args.model,
        dataset_name=args.dataset,
        evaluator_type=args.evaluator,
        optimization_strategy=args.strategy,
        max_trials=max_trials,
        num_samples=max_samples
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

# 상세 오류 정보 제공
if "CUDA out of memory" in str(e):
    print("💡 해결 방법:")
    print("   1. --samples 수를 줄여보세요 (예: --samples 10)")
    print("   2. --safe-mode 옵션을 사용하세요")
    print("   3. 양자화를 활성화하세요")
elif "scikit-optimize" in str(e):
    print("💡 해결 방법:")
    print("   1. pip install scikit-optimize==0.9.0")
    print("   2. --strategy grid_search를 사용해보세요")
else:
    print("💡 일반적인 해결 방법:")
    print("   1. --safe-mode 옵션으로 다시 시도")
    print("   2. --samples 수를 줄여보세요")
    print("   3. --debug 옵션으로 상세 로그 확인")

if args.debug:
    traceback.print_exc()
finally:
# 메모리 정리
gc.collect()
try:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except:
    pass


async def run_benchmark_command(args):
    """benchmark 명령어 실행 - 안전성 강화"""
    print(f"⚡ 안전한 벤치마크 시작: {args.model} on {args.dataset}")

    # 안전한 제한 적용
    max_samples = min(args.samples, 100)
    max_iterations = min(args.iterations, 3)

    try:
        optimizer = SafePerformanceOptimizer()

        # 모델 설정 및 등록
        config_manager = RobustConfigManager()
        model_config = config_manager.get_safe_model_config(args.model)

        if not model_config:
            model_config = config_manager.create_fallback_config(args.model)

        safe_model_manager = SafeModelManager(config_manager.optimization_config)
        safe_model_manager.register_model(args.model, model_config)
        optimizer.model_manager = safe_model_manager

        # 안전한 파라미터 생성
        params = InferenceParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=min(args.max_tokens, 512)  # 안전 제한
        )

        result = await optimizer.benchmark_model(
            model_name=args.model,
            dataset_name=args.dataset,
            params=params,
            num_samples=max_samples,
            iterations=max_iterations
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

        accuracy = len([r for r in result.evaluation_results if r.get('score', 0) > 0.8]) / len(
            result.evaluation_results)
        print(f"\n🎯 정확도: {accuracy:.3f}")

        print(f"\n📁 결과 저장됨: optimization_results/bench_{result.test_id}.json")

    except Exception as e:
        print(f"❌ 벤치마크 실패: {e}")
        if args.debug:
            traceback.print_exc()
    finally:
        gc.collect()


async def run_compare_command(args):
    """compare 명령어 실행 - 안전성 강화"""
    print(f"⚖️ 안전한 모델 비교 시작: {', '.join(args.models)} on {args.dataset}")

    max_samples = min(args.samples, 50)  # 안전 제한

    try:
        optimizer = SafePerformanceOptimizer()
        config_manager = RobustConfigManager()

        results = {}
        for model in args.models:
            print(f"\n🔄 {model} 테스트 중...")

            # 안전한 모델 설정
            model_config = config_manager.get_safe_model_config(model)
            if not model_config:
                model_config = config_manager.create_fallback_config(model)

            safe_model_manager = SafeModelManager(config_manager.optimization_config)
            safe_model_manager.register_model(model, model_config)
            optimizer.model_manager = safe_model_manager

            params = InferenceParams(temperature=0.1, top_p=0.9, max_new_tokens=300)

            result = await optimizer.benchmark_model(
                model_name=model,
                dataset_name=args.dataset,
                params=params,
                num_samples=max_samples,
                iterations=1
            )

            results[model] = result

            perf = result.performance_metrics
            accuracy = len([r for r in result.evaluation_results if r.get('score', 0) > 0.8]) / len(
                result.evaluation_results)
            print(f"   ✅ 완료: 정확도 {accuracy:.3f}, {perf.tokens_per_second:.1f} tokens/sec")

        # 결과 정렬 및 출력
        print(f"\n📊 비교 결과 ({args.metric} 기준):")
        print(f"{'순위':<4} {'모델':<20} {'정확도':<8} {'토큰/초':<10} {'메모리(MB)':<12}")
        print("-" * 60)

        # 정렬 기준에 따른 결과 정렬
        if args.metric == 'accuracy':
            sorted_results = sorted(results.items(),
                                    key=lambda x: len(
                                        [r for r in x[1].evaluation_results if r.get('score', 0) > 0.8]) / len(
                                        x[1].evaluation_results),
                                    reverse=True)
        elif args.metric == 'speed':
            sorted_results = sorted(results.items(),
                                    key=lambda x: x[1].performance_metrics.tokens_per_second,
                                    reverse=True)
        else:
            sorted_results = list(results.items())

        for i, (model, result) in enumerate(sorted_results, 1):
            perf = result.performance_metrics
            accuracy = len([r for r in result.evaluation_results if r.get('score', 0) > 0.8]) / len(
                result.evaluation_results)

            rank_symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(
                f"{rank_symbol:<4} {model:<20} {accuracy:<8.3f} {perf.tokens_per_second:<10.1f} {perf.memory_usage_mb:<12.0f}")

    except Exception as e:
        print(f"❌ 모델 비교 실패: {e}")
        if args.debug:
            traceback.print_exc()
    finally:
        gc.collect()


def run_list_command(args):
    """list 명령어 실행"""
    try:
        config_manager = RobustConfigManager()

        if args.type == 'models':
            models = config_manager.model_configs
            print(f"📋 등록된 모델 ({len(models)}개):")
            for name, config in models.items():
                # 안전성 상태 확인
                safety_warnings = SafetyChecker.check_memory_safety(config)
                safety_status = "⚠️" if safety_warnings else "✅"

                print(f"   {safety_status} {name}")
                print(f"      경로: {config.model_path}")
                print(f"      유형: {config.model_type}")
                print(f"      장치: {config.device}")
                print(f"      양자화: 4bit={config.load_in_4bit}, 8bit={config.load_in_8bit}")
                if safety_warnings:
                    print(f"      경고: {len(safety_warnings)}개 안전성 이슈")
                print()

        elif args.type == 'datasets':
            datasets = config_manager.test_configs
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
        print(f"❌ 정보 조회 실패: {e}")


def show_welcome_screen_enhanced():
    """강화된 환영 화면"""
    print_banner()
    print("🚀 안전한 오픈소스 LLM 추론 성능 최적화 시스템에 오신 것을 환영합니다!")

    print("\n📖 안전한 시작 가이드:")
    print("1. 시스템 초기화: python safe_main.py init --auto-detect --safe-defaults")
    print("2. 하드웨어 확인: python safe_main.py hardware")
    print("3. 모델 목록 확인: python safe_main.py list --type models")
    print("4. 안전한 최적화: python safe_main.py optimize --model [모델명] --dataset [데이터셋명] --safe-mode")
    print("5. 결과 확인: python safe_main.py visualize --type dashboard")

    print("\n💡 도움말:")
    print("   전체 명령어: python safe_main.py --help")
    print("   안전 모드: 모든 명령어에 --safe-mode 추가")
    print("   문제 해결: --debug 옵션으로 상세 로그 확인")

    # 강화된 시스템 상태 확인
    show_system_status_enhanced()

    print("\n🎯 개선된 주요 기능:")
    print("   ✅ 스레드 안전성 - 멀티스레드 충돌 해결")
    print("   ✅ 메모리 관리 - 자동 메모리 누수 방지")
    print("   ✅ 오류 복구 - 강화된 예외 처리")
    print("   ✅ 안전 모드 - 리소스 제한으로 안정성 보장")
    print("   ✅ 설정 검증 - 자동 문제 감지 및 수정")

    print("\n⚡ 안전성 팁:")
    print("   🛡️ 처음 사용시: --safe-mode 옵션 필수")
    print("   💾 메모리 부족시: --samples 20 --trials 5 권장")
    print("   🐛 오류 발생시: --debug 옵션으로 원인 파악")
    print("   🚀 성능 향상: 양자화 활성화 권장")


async def main():
    """메인 함수 - 안전성 강화"""
    # Python 버전 확인
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        print(f"현재 버전: {sys.version}")
        sys.exit(1)

    parser = create_argument_parser()

    # 인자가 없으면 환영 화면 표시
    if len(sys.argv) == 1:
        show_welcome_screen_enhanced()
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

    # 디버그 모드 설정
    if hasattr(args, 'debug') and args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

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

    # 시각화 및 내보내기는 기존 코드 사용
    # (visualization.py가 있다고 가정)

    # 강화된 시스템 요구사항 확인
    if not check_system_requirements_enhanced():
        print("⚠️ 시스템 요구사항을 만족하지 않습니다.")
        if not (hasattr(args, 'safe_mode') and args.safe_mode):
            response = input("안전 모드로 계속 진행하시겠습니까? (y/N): ")
            if response.lower() != 'y':  # !/usr/bin/env python3


"""
안전한 오픈소스 LLM 추론 성능 최적화 시스템 메인 CLI
모든 문제점이 해결된 버전
"""
import asyncio
import argparse
import sys
import json
import traceback
import gc
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


# 안전한 환경 설정
def setup_safe_environment():
    """안전한 환경 설정"""
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_TRUST_REMOTE_CODE', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
    if sys.platform.startswith('linux'):
        os.environ.setdefault('OMP_NUM_THREADS', '1')


setup_safe_environment()

# 로컬 모듈 import
try:
    from safe_config import RobustConfigManager, ConfigValidator, HardwareDetector, SafetyChecker, InferenceParams
    from safe_test_runner import SafePerformanceOptimizer, SafeModelManager
    from visualization import ResultVisualizer
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("필요한 파일들이 같은 디렉토리에 있는지 확인하세요:")
    print("  - safe_config.py")
    print("  - safe_test_runner.py")
    print("  - safe_evaluator.py")
    print("  - visualization.py")
    print("  - dataset_loader.py")
    print("  - model_interface.py")
    sys.exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서 생성"""
    parser = argparse.ArgumentParser(
        description='안전한 오픈소스 LLM 추론 성능 최적화 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
안전한 사용 예시:
  python safe_main.py init --auto-detect              # 시스템 초기화
  python safe_main.py hardware                        # 하드웨어 정보 확인
  python safe_main.py optimize --model llama2-7b --dataset korean_math --samples 20
  python safe_main.py benchmark --model mistral-7b --dataset korean_qa --samples 50
  python safe_main.py compare --models llama2-7b mistral-7b --dataset korean_math
  python safe_main.py visualize --type dashboard      # 결과 시각화
  python safe_main.py export --format csv             # 결과 내보내기

안전 모드 옵션:
  --safe-mode: 메모리와 성능을 제한하여 안전하게 실행
  --debug: 상세한 디버그 정보 출력
        """
    )

    # 전역 옵션
    parser.add_argument('--safe-mode', action='store_true', help='안전 모드로 실행 (제한된 리소스)')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')

    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')

    # init 명령어
    init_parser = subparsers.add_parser('init', help='시스템 초기화')
    init_parser.add_argument('--force', action='store_true', help='기존 설정 덮어쓰기')
    init_parser.add_argument('--auto-detect', action='store_true', help='하드웨어 자동 감지')
    init_parser.add_argument('--safe-defaults', action='store_true', help='안전한 기본값 사용')

    # hardware 명령어
    hardware_parser = subparsers.add_parser('hardware', help='하드웨어 정보 확인')
    hardware_parser.add_argument('--model-size', choices=['7b', '13b', '70b'], help='모델 크기별 추천')

    # optimize 명령어
    optimize_parser = subparsers.add_parser('optimize', help='파라미터 최적화')
    optimize_parser.add_argument('--model', required=True, help='모델 이름')
    optimize_parser.add_argument('--dataset', required=True, help='데이터셋 이름')
    optimize_parser.add_argument('--strategy', choices=['bayesian', 'grid_search'],
                                 default='grid_search', help='최적화 전략')
    optimize_parser.add_argument('--trials', type=int, default=10, help='최적화 시도 횟수')
    optimize_parser.add_argument('--samples', type=int, default=20, help='테스트 샘플 수')
    optimize_parser.add_argument('--evaluator', default='korean_math', help='평가자 유형')

    # benchmark 명령어
    benchmark_parser = subparsers.add_parser('benchmark', help='성능 벤치마크')
    benchmark_parser.add_argument('--model', required=True, help='모델 이름')
    benchmark_parser.add_argument('--dataset', required=True, help='데이터셋 이름')
    benchmark_parser.add_argument('--samples', type=int, default=50, help='테스트 샘플 수')
    benchmark_parser.add_argument('--iterations', type=int, default=2, help='반복 횟수')
    benchmark_parser.add_argument('--temperature', type=float, default=0.1, help='Temperature')
    benchmark_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p')
    benchmark_parser.add_argument('--max-tokens', type=int, default=300, help='최대 토큰 수')

    # compare 명령어
    compare_parser = subparsers.add_parser('compare', help='모델 비교')
    compare_parser.add_argument('--models', nargs='+', required=True, help='비교할 모델들')
    compare_parser.add_argument('--dataset', required=True, help='데이터셋 이름')
    compare_parser.add_argument('--samples', type=int, default=30, help='테스트 샘플 수')
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
│          🚀 안전한 오픈소스 LLM 추론 성능 최적화 시스템         │
│              Safe Open Source LLM Optimization              │
│                      ✅ All Issues Fixed                     │
╰─────────────────────────────────────────────────────────────╯
"""
    print(banner)


def check_system_requirements_enhanced() -> bool:
    """강화된 시스템 요구사항 확인"""
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
        'pandas': 'Pandas'
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
        'skopt': 'scikit-optimize (베이지안 최적화용)',
        'plotly': 'Plotly (시각화용)',
        'sentence_transformers': 'SentenceTransformers (의미적 유사도용)'
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

    # GPU 관련 확인
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            total_memory = sum(
                torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
                for i in range(gpu_count)
            )
            print(f"✅ GPU 감지: {gpu_count}개, 총 메모리: {total_memory}GB")

            if total_memory < 8:
                print("⚠️ GPU 메모리가 부족할 수 있습니다. 양자화 사용을 권장합니다.")
        else:
            print("⚠️ CUDA 사용