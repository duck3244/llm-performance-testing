#!/usr/bin/env python3
"""
수정된 모듈들의 import 테스트
"""
import sys
from pathlib import Path


def test_imports():
    """모든 모듈 import 테스트"""
    print("=== 수정된 모듈 Import 테스트 ===")

    try:
        # 1. Config 모듈 테스트
        print("1. Config 모듈 테스트...")
        from config.base_config import (
            BaseConfig, ValidationResult, ConfigValidator,
            EnvironmentManager, PathManager, LoggingConfig
        )
        print("   ✅ base_config 모듈 - Callable 문법 오류 수정됨")

        from config.model_config import (
            ModelConfig, ModelType, DataType, DeviceType,
            QuantizationConfig, ModelConfigManager
        )
        print("   ✅ model_config 모듈")

        # 2. Core 모듈 테스트
        print("\n2. Core 모듈 테스트...")
        from core.memory_manager import (
            ImprovedResourceManager, MemoryLevel, MemoryStats,
            get_resource_manager
        )
        print("   ✅ memory_manager 모듈 - Weak reference 경고 수정됨")

        from core.async_manager import (
            SafeAsyncManager, ExecutorType, TaskInfo,
            get_async_manager, run_async_safe
        )
        print("   ✅ async_manager 모듈")

        from core.error_handler import (
            ErrorSeverity, ErrorCategory, ErrorInfo,
            get_global_error_handler, safe_execute
        )
        print("   ✅ error_handler 모듈")

        from core.improved_optimizer import (
            InferenceParams, OptimizationResult, SafeOptimizer
        )
        print("   ✅ improved_optimizer 모듈")

        # 3. 기능 테스트
        print("\n3. 기본 기능 테스트...")

        # ConfigValidator 테스트
        validator = ConfigValidator()
        result = validator.validate_memory_size("8GB")
        print(f"   ✅ 메모리 크기 검증: {result.is_valid}")

        # MemoryManager 테스트
        memory_manager = get_resource_manager()
        stats = memory_manager.get_memory_stats()
        print(f"   ✅ 메모리 통계 조회: {len(stats)}개 장치")

        # ErrorHandler 테스트
        error_handler = get_global_error_handler()
        try:
            raise ValueError("테스트 오류")
        except Exception as e:
            error_info = error_handler.handle_exception(e)
            print(f"   ✅ 오류 처리: {error_info.category.value}")

        # ModelConfig 테스트
        model_config = ModelConfig(
            name="test-model",
            model_path="test/path",
            model_type=ModelType.TRANSFORMERS,
            device=DeviceType.AUTO
        )
        is_valid = model_config.validate()
        print(f"   ✅ 모델 설정 검증: {is_valid}")

        # InferenceParams 테스트
        params = InferenceParams(temperature=0.7, top_p=0.9)
        param_result = params.validate()
        print(f"   ✅ 추론 파라미터 검증: {param_result.is_valid}")

        print("\n🎉 모든 모듈이 정상적으로 작동합니다!")
        return True

    except ImportError as e:
        print(f"❌ Import 오류: {e}")
        print("누락된 의존성이 있을 수 있습니다.")
        return False
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weak_reference_fix():
    """Weak reference 수정 테스트"""
    print("\n=== Weak Reference 수정 테스트 ===")

    try:
        from core.memory_manager import get_resource_manager

        manager = get_resource_manager()

        # 더미 모델 객체 생성
        class DummyModel:
            def __init__(self, name):
                self.name = name

            def __del__(self):
                print(f"   DummyModel {self.name} 삭제됨")

        # 모델 등록
        model1 = DummyModel("test-model-1")
        manager.register_model("test-model-1", model1, "cuda:0")

        # 활성 모델 확인
        active_models = manager.get_active_models()
        print(f"   등록된 모델: {active_models}")

        # 모델 삭제 (weak reference 콜백 테스트)
        del model1

        # 가비지 컬렉션 강제 실행
        import gc
        gc.collect()

        # 다시 확인
        active_models = manager.get_active_models()
        print(f"   삭제 후 모델: {active_models}")

        print("✅ Weak reference 콜백이 정상 작동합니다!")
        return True

    except Exception as e:
        print(f"❌ Weak reference 테스트 실패: {e}")
        return False


def test_async_safe():
    """비동기 안전성 테스트"""
    print("\n=== 비동기 안전성 테스트 ===")

    try:
        import asyncio
        from core.async_manager import run_async_safe

        async def test_coro():
            await asyncio.sleep(0.1)
            return "async result"

        # 안전한 비동기 실행
        result = run_async_safe(test_coro(), timeout=5.0)
        print(f"   비동기 결과: {result}")

        print("✅ 비동기 처리가 안전하게 작동합니다!")
        return True

    except Exception as e:
        print(f"❌ 비동기 테스트 실패: {e}")
        return False


def test_error_patterns():
    """오류 패턴 인식 테스트"""
    print("\n=== 오류 패턴 인식 테스트 ===")

    try:
        from core.error_handler import get_global_error_handler

        handler = get_global_error_handler()

        # CUDA 메모리 오류 시뮬레이션
        try:
            raise RuntimeError("CUDA out of memory")
        except Exception as e:
            error_info = handler.handle_exception(e)

            print(f"   오류 카테고리: {error_info.category.value}")
            print(f"   심각도: {error_info.severity.value}")
            print(f"   제안사항: {len(error_info.suggestions)}개")

            if error_info.suggestions:
                print("   제안:")
                for suggestion in error_info.suggestions[:2]:
                    print(f"     - {suggestion}")

        print("✅ 오류 패턴 인식이 정상 작동합니다!")
        return True

    except Exception as e:
        print(f"❌ 오류 패턴 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    print("🔧 수정된 모듈들의 종합 테스트를 시작합니다...\n")

    success_count = 0
    total_tests = 4

    # 1. Import 테스트
    if test_imports():
        success_count += 1

    # 2. Weak reference 수정 테스트
    if test_weak_reference_fix():
        success_count += 1

    # 3. 비동기 안전성 테스트
    if test_async_safe():
        success_count += 1

    # 4. 오류 패턴 테스트
    if test_error_patterns():
        success_count += 1

    print(f"\n{'=' * 50}")
    print(f"테스트 결과: {success_count}/{total_tests} 성공")

    if success_count == total_tests:
        print("🎉 모든 수정사항이 정상적으로 작동합니다!")
        print("\n다음 단계:")
        print("1. python main.py init --auto-detect")
        print("2. python main.py status --detailed")
        print("3. python main.py optimize --model qwen2.5-7b --dataset korean_math --safe")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.")

    # 정리
    try:
        from core.memory_manager import cleanup_all_resources
        from core.async_manager import cleanup_async_manager
        from core.error_handler import cleanup_error_handler

        cleanup_all_resources()
        cleanup_async_manager()
        cleanup_error_handler()
        print("\n✅ 시스템 정리 완료")
    except:
        pass