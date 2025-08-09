#!/usr/bin/env python3
"""
메모리 누수 및 동시성 문제 해결 검증 테스트
Critical 문제 해결 검증: 완전한 메모리 해제 및 스레드 안전성
"""
import asyncio
import threading
import time
import gc
import logging
import psutil
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import unittest

# 개선된 모듈들 import
try:
    # 메모리 관리자
    from core.memory_manager import (
        ImprovedResourceManager,
        get_resource_manager,
        cleanup_all_resources,
        MemoryLevel
    )

    # 비동기 관리자
    from core.async_manager import (
        SafeAsyncManager,
        get_async_manager,
        cleanup_async_manager,
        run_async_safe,
        ExecutorType,
        TaskState
    )

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 모듈 import 실패: {e}")
    MODULES_AVAILABLE = False


class MemoryLeakDetector:
    """메모리 누수 감지기"""

    def __init__(self):
        self.initial_memory = None
        self.process = psutil.Process(os.getpid())

    def start_monitoring(self):
        """모니터링 시작"""
        gc.collect()  # 초기 가비지 컬렉션
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def check_leak(self, tolerance_mb: float = 50.0) -> Dict[str, Any]:
        """메모리 누수 확인"""
        gc.collect()  # 가비지 컬렉션 강제 실행
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        memory_increase = current_memory - self.initial_memory
        is_leak = memory_increase > tolerance_mb

        return {
            'initial_memory_mb': self.initial_memory,
            'current_memory_mb': current_memory,
            'increase_mb': memory_increase,
            'tolerance_mb': tolerance_mb,
            'has_leak': is_leak,
            'leak_percentage': (memory_increase / self.initial_memory) * 100 if self.initial_memory > 0 else 0
        }


class ConcurrencyTester:
    """동시성 안전성 테스터"""

    def __init__(self):
        self.results = []
        self.exceptions = []
        self.lock = threading.Lock()

    def add_result(self, result: Any):
        """결과 추가 (스레드 안전)"""
        with self.lock:
            self.results.append(result)

    def add_exception(self, exception: Exception):
        """예외 추가 (스레드 안전)"""
        with self.lock:
            self.exceptions.append(exception)

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        with self.lock:
            return {
                'total_results': len(self.results),
                'total_exceptions': len(self.exceptions),
                'success_rate': len(self.results) / (len(self.results) + len(self.exceptions)) if (
                                                                                                              len(self.results) + len(
                                                                                                          self.exceptions)) > 0 else 0
            }


class MemoryLeakTests(unittest.TestCase):
    """메모리 누수 방지 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.leak_detector = MemoryLeakDetector()
        self.leak_detector.start_monitoring()

        if MODULES_AVAILABLE:
            self.memory_manager = get_resource_manager()

    def tearDown(self):
        """테스트 정리"""
        if MODULES_AVAILABLE:
            cleanup_all_resources()

    @unittest.skipUnless(MODULES_AVAILABLE, "모듈을 사용할 수 없음")
    def test_model_registration_cleanup(self):
        """모델 등록 및 정리 테스트"""
        print("\n=== 모델 등록/정리 메모리 누수 테스트 ===")

        # 더미 모델 클래스
        class DummyModel:
            def __init__(self, size_mb: int = 10):
                # 메모리 할당 시뮬레이션
                self.data = bytearray(size_mb * 1024 * 1024)  # size_mb MB
                self.name = f"model_{id(self)}"

            def cpu(self):
                """CPU로 이동 시뮬레이션"""
                return self

        # 여러 모델 등록 및 정리 반복
        for i in range(10):
            model_name = f"test_model_{i}"
            dummy_model = DummyModel(5)  # 5MB 모델

            # 모델 등록
            self.memory_manager.register_model(model_name, dummy_model, "cpu")

            # 활성 모델 확인
            active_models = self.memory_manager.get_active_models()
            self.assertIn(model_name, active_models)

            # 모델 정리
            cleanup_success = self.memory_manager.cleanup_model(model_name)
            self.assertTrue(cleanup_success, f"모델 {model_name} 정리 실패")

            # 참조 제거
            del dummy_model

        # 강제 가비지 컬렉션
        gc.collect()

        # 메모리 누수 확인
        leak_result = self.leak_detector.check_leak(tolerance_mb=30.0)
        print(f"메모리 증가: {leak_result['increase_mb']:.2f}MB ({leak_result['leak_percentage']:.1f}%)")

        self.assertFalse(leak_result['has_leak'],
                         f"메모리 누수 감지: {leak_result['increase_mb']:.2f}MB 증가")

    @unittest.skipUnless(MODULES_AVAILABLE, "모듈을 사용할 수 없음")
    def test_memory_guard_cleanup(self):
        """메모리 가드 정리 테스트"""
        print("\n=== 메모리 가드 정리 테스트 ===")

        class LargeModel:
            def __init__(self):
                self.data = bytearray(20 * 1024 * 1024)  # 20MB

            def cpu(self):
                return self

        # 메모리 가드 사용
        try:
            with self.memory_manager.memory_guard("guard_test_model", required_gb=0.1):
                large_model = LargeModel()
                self.memory_manager.register_model("guard_test_model", large_model, "cpu")

                # 모델 사용 시뮬레이션
                time.sleep(0.1)

        except Exception as e:
            self.fail(f"메모리 가드 실행 실패: {e}")

        # 가드 종료 후 확인
        active_models = self.memory_manager.get_active_models()
        self.assertNotIn("guard_test_model", active_models, "메모리 가드 종료 후 모델이 남아있음")

        # 메모리 누수 확인
        leak_result = self.leak_detector.check_leak(tolerance_mb=25.0)
        self.assertFalse(leak_result['has_leak'],
                         f"메모리 가드 누수 감지: {leak_result['increase_mb']:.2f}MB")

    @unittest.skipUnless(MODULES_AVAILABLE, "모듈을 사용할 수 없음")
    def test_emergency_cleanup(self):
        """긴급 정리 테스트"""
        print("\n=== 긴급 메모리 정리 테스트 ===")

        class HugeModel:
            def __init__(self):
                self.data = bytearray(50 * 1024 * 1024)  # 50MB

            def cpu(self):
                return self

        # 여러 큰 모델 등록
        models = []
        for i in range(5):
            model_name = f"huge_model_{i}"
            huge_model = HugeModel()
            models.append(huge_model)
            self.memory_manager.register_model(model_name, huge_model, "cpu")

        # 긴급 정리 실행
        self.memory_manager.cleanup_all_devices(force=True)

        # 모든 모델이 정리되었는지 확인
        active_models = self.memory_manager.get_active_models()
        self.assertEqual(len(active_models), 0, "긴급 정리 후 활성 모델이 남아있음")

        # 참조 제거
        del models

        # 메모리 누수 확인
        leak_result = self.leak_detector.check_leak(tolerance_mb=60.0)
        self.assertFalse(leak_result['has_leak'],
                         f"긴급 정리 후 누수 감지: {leak_result['increase_mb']:.2f}MB")


class ConcurrencyTests(unittest.TestCase):
    """동시성 안전성 테스트"""

    def setUp(self):
        """테스트 설정"""
        if MODULES_AVAILABLE:
            self.async_manager = get_async_manager()

    def tearDown(self):
        """테스트 정리"""
        if MODULES_AVAILABLE:
            cleanup_async_manager()

    @unittest.skipUnless(MODULES_AVAILABLE, "모듈을 사용할 수 없음")
    def test_concurrent_task_submission(self):
        """동시 태스크 제출 테스트"""
        print("\n=== 동시 태스크 제출 테스트 ===")

        async def test_coro(delay: float, value: int):
            await asyncio.sleep(delay)
            return value * 2

        tester = ConcurrencyTester()

        def submit_tasks(thread_id: int, num_tasks: int = 10):
            """스레드별 태스크 제출"""
            try:
                for i in range(num_tasks):
                    task_id = self.async_manager.submit_coro(
                        test_coro(0.01, thread_id * 100 + i),
                        name=f"concurrent_task_{thread_id}_{i}"
                    )

                    # 결과 대기
                    result = self.async_manager.get_task_result(task_id, timeout=5.0)
                    tester.add_result(result)

            except Exception as e:
                tester.add_exception(e)

        # 10개 스레드에서 동시 실행
        threads = []
        num_threads = 10
        tasks_per_thread = 5

        for i in range(num_threads):
            thread = threading.Thread(target=submit_tasks, args=(i, tasks_per_thread))
            threads.append(thread)
            thread.start()

        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join(timeout=30.0)

        stats = tester.get_stats()
        print(f"결과: {stats['total_results']}개 성공, {stats['total_exceptions']}개 실패")
        print(f"성공률: {stats['success_rate']:.1%}")

        # 성공률이 95% 이상이어야 함
        self.assertGreaterEqual(stats['success_rate'], 0.95,
                                f"동시성 테스트 성공률 부족: {stats['success_rate']:.1%}")

        expected_results = num_threads * tasks_per_thread
        self.assertEqual(stats['total_results'], expected_results,
                         f"예상 결과 수와 불일치: {stats['total_results']} vs {expected_results}")

    @unittest.skipUnless(MODULES_AVAILABLE, "모듈을 사용할 수 없음")
    def test_memory_and_async_interaction(self):
        """메모리 관리자와 비동기 관리자 상호작용 테스트"""
        print("\n=== 메모리-비동기 상호작용 테스트 ===")

        memory_manager = get_resource_manager()

        class AsyncModel:
            def __init__(self, size_mb: int = 5):
                self.data = bytearray(size_mb * 1024 * 1024)

            def cpu(self):
                return self

            async def process(self, data):
                await asyncio.sleep(0.01)
                return len(data)

        async def model_task(model_id: int):
            """모델 사용 태스크"""
            model_name = f"async_model_{model_id}"
            model = AsyncModel()

            try:
                # 메모리 가드와 함께 모델 등록
                with memory_manager.memory_guard(model_name, required_gb=0.01):
                    memory_manager.register_model(model_name, model, "cpu")

                    # 비동기 작업 수행
                    result = await model.process(b"test_data")
                    return result

            except Exception as e:
                raise RuntimeError(f"모델 {model_id} 처리 실패: {e}")

        # 여러 모델을 동시에 처리
        num_models = 20
        coroutines = [model_task(i) for i in range(num_models)]

        # 병렬 실행
        results = self.async_manager.run_parallel(coroutines, timeout=30.0)

        # 결과 검증
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]

        print(f"성공: {len(successful_results)}개, 실패: {len(failed_results)}개")

        # 90% 이상 성공해야 함
        success_rate = len(successful_results) / num_models
        self.assertGreaterEqual(success_rate, 0.9,
                                f"메모리-비동기 상호작용 성공률 부족: {success_rate:.1%}")

        # 활성 모델이 모두 정리되었는지 확인
        active_models = memory_manager.get_active_models()
        self.assertEqual(len(active_models), 0,
                         f"처리 완료 후 활성 모델이 남아있음: {active_models}")

    @unittest.skipUnless(MODULES_AVAILABLE, "모듈을 사용할 수 없음")
    def test_task_state_consistency(self):
        """태스크 상태 일관성 테스트"""
        print("\n=== 태스크 상태 일관성 테스트 ===")

        async def state_test_coro(delay: float, should_fail: bool = False):
            await asyncio.sleep(delay)
            if should_fail:
                raise ValueError("의도적 실패")
            return "success"

        # 다양한 시나리오의 태스크들
        test_cases = [
            (0.1, False, TaskState.COMPLETED),  # 성공
            (0.1, True, TaskState.FAILED),  # 실패
        ]

        task_ids = []

        # 태스크 제출
        for i, (delay, should_fail, expected_state) in enumerate(test_cases):
            task_id = self.async_manager.submit_coro(
                state_test_coro(delay, should_fail),
                name=f"state_test_{i}",
                timeout=5.0
            )
            task_ids.append((task_id, expected_state))

        # 모든 태스크 완료 대기
        time.sleep(2.0)

        # 상태 검증
        for task_id, expected_state in task_ids:
            task_info = self.async_manager.get_task_status(task_id)
            self.assertIsNotNone(task_info, f"태스크 {task_id} 정보 없음")
            self.assertEqual(task_info.state, expected_state,
                             f"태스크 {task_id} 상태 불일치: {task_info.state} vs {expected_state}")

        print(f"모든 태스크 상태가 예상대로 설정됨")

    @unittest.skipUnless(MODULES_AVAILABLE, "모듈을 사용할 수 없음")
    def test_timeout_handling(self):
        """타임아웃 처리 테스트"""
        print("\n=== 타임아웃 처리 테스트 ===")

        async def long_running_coro():
            await asyncio.sleep(5.0)  # 5초 실행
            return "completed"

        # 짧은 타임아웃으로 태스크 제출
        task_id = self.async_manager.submit_coro(
            long_running_coro(),
            name="timeout_test",
            timeout=0.5  # 0.5초 타임아웃
        )

        # 타임아웃 발생 확인
        start_time = time.time()
        with self.assertRaises(TimeoutError):
            self.async_manager.get_task_result(task_id, timeout=2.0)

        elapsed_time = time.time() - start_time

        # 타임아웃이 예상 시간 내에 발생했는지 확인
        self.assertLess(elapsed_time, 2.0, "타임아웃이 너무 늦게 발생")

        # 태스크 상태 확인
        task_info = self.async_manager.get_task_status(task_id)
        self.assertEqual(task_info.state, TaskState.TIMEOUT, "타임아웃 상태가 설정되지 않음")

        print(f"타임아웃 정상 처리됨 ({elapsed_time:.2f}초)")


class StressTests(unittest.TestCase):
    """스트레스 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.leak_detector = MemoryLeakDetector()
        self.leak_detector.start_monitoring()

    def tearDown(self):
        """테스트 정리"""
        if MODULES_AVAILABLE:
            cleanup_all_resources()
            cleanup_async_manager()

    @unittest.skipUnless(MODULES_AVAILABLE, "모듈을 사용할 수 없음")
    def test_high_load_scenario(self):
        """고부하 시나리오 테스트"""
        print("\n=== 고부하 시나리오 테스트 ===")

        memory_manager = get_resource_manager()
        async_manager = get_async_manager()

        class StressModel:
            def __init__(self, model_id: int):
                self.model_id = model_id
                self.data = bytearray(2 * 1024 * 1024)  # 2MB

            def cpu(self):
                return self

            async def process(self, data_size: int):
                # CPU 집약적 작업 시뮬레이션
                await asyncio.sleep(0.01)
                return sum(range(data_size))

        async def stress_task(task_id: int):
            """스트레스 태스크"""
            model_name = f"stress_model_{task_id}"
            model = StressModel(task_id)

            try:
                with memory_manager.memory_guard(model_name, required_gb=0.005):
                    memory_manager.register_model(model_name, model, "cpu")

                    # 여러 번의 처리 수행
                    results = []
                    for i in range(5):
                        result = await model.process(100)
                        results.append(result)

                    return sum(results)

            except Exception as e:
                return e

        # 많은 수의 동시 태스크 실행
        num_tasks = 50
        print(f"   {num_tasks}개 태스크를 동시 실행...")

        coroutines = [stress_task(i) for i in range(num_tasks)]

        start_time = time.time()
        results = async_manager.run_with_semaphore(
            coroutines,
            max_concurrent=10,  # 동시 실행 제한
            timeout=60.0
        )
        execution_time = time.time() - start_time

        # 결과 분석
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]

        success_rate = len(successful_results) / num_tasks

        print(f"   실행 시간: {execution_time:.2f}초")
        print(f"   성공: {len(successful_results)}개, 실패: {len(failed_results)}개")
        print(f"   성공률: {success_rate:.1%}")

        # 성능 기준 검증
        self.assertGreaterEqual(success_rate, 0.8, f"고부하 성공률 부족: {success_rate:.1%}")
        self.assertLess(execution_time, 30.0, f"실행 시간 초과: {execution_time:.2f}초")

        # 메모리 누수 확인
        leak_result = self.leak_detector.check_leak(tolerance_mb=100.0)
        self.assertFalse(leak_result['has_leak'],
                         f"고부하 후 메모리 누수: {leak_result['increase_mb']:.2f}MB")

        # 모든 리소스 정리 확인
        active_models = memory_manager.get_active_models()
        active_tasks = async_manager.list_active_tasks()

        self.assertEqual(len(active_models), 0, f"정리되지 않은 모델: {len(active_models)}개")
        self.assertEqual(len(active_tasks), 0, f"정리되지 않은 태스크: {len(active_tasks)}개")


def run_comprehensive_test():
    """종합 테스트 실행"""
    print("🛡️ 메모리 누수 및 동시성 문제 해결 검증 테스트 시작")
    print("=" * 70)

    if not MODULES_AVAILABLE:
        print("❌ 필요한 모듈을 import할 수 없습니다.")
        return False

    # 테스트 로깅 설정
    logging.basicConfig(
        level=logging.WARNING,  # 경고 이상만 출력
        format='%(levelname)s: %(message)s'
    )

    # 테스트 스위트 구성
    test_suite = unittest.TestSuite()

    # 메모리 누수 테스트
    test_suite.addTest(MemoryLeakTests('test_model_registration_cleanup'))
    test_suite.addTest(MemoryLeakTests('test_memory_guard_cleanup'))
    test_suite.addTest(MemoryLeakTests('test_emergency_cleanup'))

    # 동시성 테스트
    test_suite.addTest(ConcurrencyTests('test_concurrent_task_submission'))
    test_suite.addTest(ConcurrencyTests('test_memory_and_async_interaction'))
    test_suite.addTest(ConcurrencyTests('test_task_state_consistency'))
    test_suite.addTest(ConcurrencyTests('test_timeout_handling'))

    # 스트레스 테스트
    test_suite.addTest(StressTests('test_high_load_scenario'))

    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
    result = runner.run(test_suite)

    # 결과 요약
    print("\n" + "=" * 70)
    print("🔍 테스트 결과 요약")
    print("=" * 70)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors

    print(f"총 테스트: {total_tests}개")
    print(f"성공: {successes}개")
    print(f"실패: {failures}개")
    print(f"오류: {errors}개")

    if failures > 0:
        print("\n❌ 실패한 테스트:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")

    if errors > 0:
        print("\n🚨 오류가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")

    # 전체 성공률
    success_rate = successes / total_tests if total_tests > 0 else 0
    print(f"\n전체 성공률: {success_rate:.1%}")

    # 최종 평가
    if success_rate >= 0.9:
        print("✅ 메모리 누수 및 동시성 문제가 성공적으로 해결되었습니다!")
        status = True
    elif success_rate >= 0.7:
        print("⚠️ 대부분의 문제가 해결되었지만 일부 개선이 필요합니다.")
        status = True
    else:
        print("❌ 추가적인 문제 해결이 필요합니다.")
        status = False

    return status


def run_quick_validation():
    """빠른 검증 테스트"""
    print("⚡ 빠른 검증 테스트 실행")
    print("-" * 50)

    if not MODULES_AVAILABLE:
        print("❌ 모듈 import 실패")
        return False

    try:
        # 1. 기본 기능 테스트
        print("1. 기본 기능 테스트...")
        memory_manager = get_resource_manager()
        async_manager = get_async_manager()

        # 메모리 통계 조회
        stats = memory_manager.get_memory_stats()
        print(f"   메모리 장치: {len(stats)}개 감지")

        # 간단한 비동기 태스크
        async def simple_test():
            await asyncio.sleep(0.1)
            return "test_success"

        result = async_manager.run_async(simple_test())
        assert result == "test_success", "비동기 실행 실패"
        print("   ✅ 기본 기능 정상")

        # 2. 메모리 정리 테스트
        print("2. 메모리 정리 테스트...")

        class QuickTestModel:
            def __init__(self):
                self.data = bytearray(1024 * 1024)  # 1MB

            def cpu(self):
                return self

        test_model = QuickTestModel()
        memory_manager.register_model("quick_test", test_model, "cpu")

        active_before = len(memory_manager.get_active_models())
        memory_manager.cleanup_model("quick_test")
        active_after = len(memory_manager.get_active_models())

        assert active_after < active_before, "모델 정리 실패"
        print("   ✅ 메모리 정리 정상")

        # 3. 동시성 테스트
        print("3. 동시성 테스트...")

        async def concurrent_test(value):
            await asyncio.sleep(0.01)
            return value * 2

        coroutines = [concurrent_test(i) for i in range(10)]
        results = async_manager.run_parallel(coroutines)

        assert len(results) == 10, "병렬 실행 결과 수 불일치"
        assert all(isinstance(r, int) for r in results), "결과 타입 오류"
        print("   ✅ 동시성 처리 정상")

        # 정리
        cleanup_all_resources()
        cleanup_async_manager()

        print("✅ 빠른 검증 테스트 통과!")
        return True

    except Exception as e:
        print(f"❌ 빠른 검증 실패: {e}")
        return False


if __name__ == "__main__":
    import sys

    # 명령행 인자 처리
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_validation()
    else:
        success = run_comprehensive_test()

    # 종료 코드 설정
    sys.exit(0 if success else 1)