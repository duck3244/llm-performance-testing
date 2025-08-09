"""
개선된 메모리 관리 시스템 - 메모리 누수 방지 및 동시성 문제 해결
Critical 문제 해결: 완전한 메모리 해제 보장 및 스레드 안전성 강화
"""
import gc
import time
import torch
import psutil
import threading
import weakref
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from enum import Enum
import contextlib
from collections import defaultdict
import atexit


class MemoryLevel(Enum):
    """메모리 위험 수준"""
    SAFE = "safe"           # < 70%
    WARNING = "warning"     # 70-85%
    CRITICAL = "critical"   # 85-95%
    EMERGENCY = "emergency" # > 95%


@dataclass
class MemoryStats:
    """메모리 통계"""
    device: str
    total_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    utilization: float
    level: MemoryLevel
    timestamp: float


class ThreadSafeModelRegistry:
    """스레드 안전한 모델 레지스트리"""

    def __init__(self):
        # 모델별 개별 락 사용 (데드락 방지)
        self._model_locks: Dict[str, threading.RLock] = {}
        self._registry_lock = threading.RLock()  # 레지스트리 자체 보호
        self._active_models: Dict[str, Dict[str, Any]] = {}
        self._cleanup_in_progress: Set[str] = set()  # 정리 중인 모델 추적

    def register_model(self, model_name: str, model_ref: Any, device: str):
        """모델 등록 (스레드 안전)"""
        with self._registry_lock:
            # 모델별 락 생성
            if model_name not in self._model_locks:
                self._model_locks[model_name] = threading.RLock()

            # 약한 참조로 모델 등록
            def cleanup_callback(ref):
                self._safe_model_cleanup(model_name)

            self._active_models[model_name] = {
                'ref': weakref.ref(model_ref, cleanup_callback),
                'device': device,
                'registered_at': time.time(),
                'cleanup_count': 0  # 정리 시도 횟수 추적
            }

    def get_model_lock(self, model_name: str) -> Optional[threading.RLock]:
        """모델별 락 반환"""
        with self._registry_lock:
            return self._model_locks.get(model_name)

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """모델 정보 조회 (스레드 안전)"""
        with self._registry_lock:
            return self._active_models.get(model_name)

    def remove_model(self, model_name: str) -> bool:
        """모델 제거 (스레드 안전)"""
        with self._registry_lock:
            if model_name in self._cleanup_in_progress:
                return False  # 이미 정리 중

            self._cleanup_in_progress.add(model_name)

            try:
                removed = model_name in self._active_models
                if removed:
                    del self._active_models[model_name]
                    # 락은 유지 (다른 스레드에서 사용 중일 수 있음)
                return removed
            finally:
                self._cleanup_in_progress.discard(model_name)

    def _safe_model_cleanup(self, model_name: str):
        """안전한 모델 정리 (weakref 콜백)"""
        try:
            self.remove_model(model_name)
        except Exception as e:
            # 콜백에서는 예외를 로깅만 하고 전파하지 않음
            logging.getLogger(__name__).warning(f"모델 {model_name} 자동 정리 실패: {e}")

    def list_active_models(self) -> List[str]:
        """활성 모델 목록 (스레드 안전)"""
        with self._registry_lock:
            active = []
            for name, info in self._active_models.items():
                if info['ref']() is not None:  # 아직 살아있는 모델만
                    active.append(name)
            return active

    def cleanup_dead_references(self) -> int:
        """죽은 참조 정리"""
        with self._registry_lock:
            dead_models = []
            for name, info in self._active_models.items():
                if info['ref']() is None:
                    dead_models.append(name)

            for name in dead_models:
                self.remove_model(name)

            return len(dead_models)


class MemoryCleanupManager:
    """메모리 정리 전담 매니저 - 재시도 로직 포함"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._device_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        self._last_cleanup_time: Dict[str, float] = {}
        self._cleanup_failure_count: Dict[str, int] = defaultdict(int)

    def cleanup_device_memory(self, device: str, max_retries: int = 3) -> bool:
        """장치별 메모리 완전 정리 (재시도 로직 포함)"""
        with self._device_locks[device]:
            for attempt in range(max_retries):
                try:
                    success = self._attempt_cleanup(device)
                    if success:
                        self._cleanup_failure_count[device] = 0
                        self._last_cleanup_time[device] = time.time()
                        return True

                except Exception as e:
                    self.logger.warning(f"장치 {device} 정리 시도 {attempt + 1}/{max_retries} 실패: {e}")

                    if attempt < max_retries - 1:
                        # 지수 백오프로 재시도
                        wait_time = min(2 ** attempt, 10)  # 최대 10초
                        time.sleep(wait_time)

            # 모든 재시도 실패
            self._cleanup_failure_count[device] += 1
            self.logger.error(f"장치 {device} 메모리 정리 완전 실패 (실패 횟수: {self._cleanup_failure_count[device]})")
            return False

    def _attempt_cleanup(self, device: str) -> bool:
        """단일 정리 시도"""
        initial_memory = self._get_device_memory_usage(device)

        if device.startswith("cuda"):
            self._cleanup_gpu_memory(device)
        elif device == "cpu":
            self._cleanup_cpu_memory()
        else:
            self.logger.warning(f"알 수 없는 장치 타입: {device}")
            return False

        # 정리 효과 검증
        final_memory = self._get_device_memory_usage(device)

        # 메모리 사용량이 감소했거나 이미 낮은 경우 성공으로 간주
        memory_reduced = final_memory < initial_memory * 0.95  # 5% 이상 감소
        memory_already_low = final_memory < 0.1  # 10% 미만 사용

        success = memory_reduced or memory_already_low

        if success:
            self.logger.debug(f"장치 {device} 메모리 정리 성공: {initial_memory:.1%} -> {final_memory:.1%}")
        else:
            self.logger.warning(f"장치 {device} 메모리 정리 효과 없음: {initial_memory:.1%} -> {final_memory:.1%}")

        return success

    def _cleanup_gpu_memory(self, device: str):
        """GPU 메모리 완전 정리"""
        device_id = int(device.split(":")[-1]) if ":" in device else 0

        with torch.cuda.device(device_id):
            # 1. 현재 스트림 동기화
            torch.cuda.synchronize(device_id)

            # 2. 캐시된 메모리 해제
            torch.cuda.empty_cache()

            # 3. IPC 컬렉션 (멀티프로세싱 환경)
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()

            # 4. 메모리 풀 통계 재설정 (PyTorch 2.0+)
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats(device_id)

            # 5. 강제 가비지 컬렉션
            gc.collect()

            # 6. 다시 한번 동기화 및 캐시 정리
            torch.cuda.synchronize(device_id)
            torch.cuda.empty_cache()

    def _cleanup_cpu_memory(self):
        """CPU 메모리 정리"""
        # 다중 가비지 컬렉션 (순환 참조 완전 제거)
        for _ in range(3):
            collected = gc.collect()
            if collected == 0:
                break  # 더 이상 정리할 것이 없음

    def _get_device_memory_usage(self, device: str) -> float:
        """장치 메모리 사용률 조회"""
        try:
            if device.startswith("cuda"):
                device_id = int(device.split(":")[-1]) if ":" in device else 0
                allocated = torch.cuda.memory_allocated(device_id)
                total = torch.cuda.get_device_properties(device_id).total_memory
                return allocated / total
            elif device == "cpu":
                return psutil.virtual_memory().percent / 100.0
            else:
                return 0.0
        except Exception:
            return 0.0

    def force_emergency_cleanup(self) -> Dict[str, bool]:
        """긴급 전체 메모리 정리"""
        results = {}

        # CPU 정리
        try:
            self._cleanup_cpu_memory()
            results["cpu"] = True
        except Exception as e:
            self.logger.error(f"CPU 긴급 정리 실패: {e}")
            results["cpu"] = False

        # 모든 GPU 정리
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = f"cuda:{i}"
                try:
                    self._cleanup_gpu_memory(device)
                    results[device] = True
                except Exception as e:
                    self.logger.error(f"GPU {i} 긴급 정리 실패: {e}")
                    results[device] = False

        return results


class ImprovedResourceManager:
    """개선된 리소스 관리자 - 메모리 누수 방지 및 동시성 강화"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 컴포넌트 초기화
        self.model_registry = ThreadSafeModelRegistry()
        self.cleanup_manager = MemoryCleanupManager()

        # 설정
        self.warning_threshold = 0.70
        self.critical_threshold = 0.85
        self.emergency_threshold = 0.95

        # 모니터링
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False
        self._monitor_interval = 30.0

        # 정리 콜백
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._cleanup_lock = threading.RLock()

        # 프로그램 종료 시 자동 정리 등록
        atexit.register(self._emergency_shutdown)

    def register_model(self, model_name: str, model_ref: Any, device: str = "auto"):
        """모델 등록 및 추적 (스레드 안전)"""
        try:
            # 장치 자동 감지
            if device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

            self.model_registry.register_model(model_name, model_ref, device)
            self.logger.info(f"모델 등록됨: {model_name} on {device}")

        except Exception as e:
            self.logger.error(f"모델 등록 실패 {model_name}: {e}")
            raise

    def cleanup_model(self, model_name: str, force: bool = False) -> bool:
        """특정 모델 완전 정리 (스레드 안전)"""
        model_lock = self.model_registry.get_model_lock(model_name)
        if not model_lock:
            self.logger.warning(f"모델 {model_name}이 등록되지 않음")
            return False

        with model_lock:
            try:
                model_info = self.model_registry.get_model_info(model_name)
                if not model_info:
                    return False

                device = model_info['device']
                model_ref = model_info['ref']()

                # 1. 모델을 CPU로 이동 (GPU 메모리 해제)
                if model_ref is not None and hasattr(model_ref, 'cpu'):
                    try:
                        self.logger.debug(f"모델 {model_name}을 CPU로 이동 중...")
                        model_ref.cpu()
                    except Exception as e:
                        self.logger.warning(f"모델 {model_name} CPU 이동 실패: {e}")
                        if not force:
                            return False

                # 2. 모델 참조 명시적 제거
                if model_ref is not None:
                    try:
                        del model_ref
                    except Exception as e:
                        self.logger.warning(f"모델 참조 제거 실패: {e}")

                # 3. 레지스트리에서 제거
                removed = self.model_registry.remove_model(model_name)

                # 4. 장치 메모리 정리 (재시도 로직 포함)
                cleanup_success = self.cleanup_manager.cleanup_device_memory(device)

                if removed and cleanup_success:
                    self.logger.info(f"모델 {model_name} 완전 정리 완료")
                    return True
                else:
                    self.logger.warning(f"모델 {model_name} 정리 부분 실패 (removed={removed}, cleanup={cleanup_success})")
                    return force  # force 모드에서는 부분 실패도 성공으로 간주

            except Exception as e:
                self.logger.error(f"모델 {model_name} 정리 중 오류: {e}")
                if force:
                    # force 모드에서는 최대한 정리 시도
                    try:
                        self.model_registry.remove_model(model_name)
                        self.cleanup_manager.cleanup_device_memory(model_info['device'])
                    except:
                        pass
                    return True
                return False

    def cleanup_all_devices(self, force: bool = False):
        """모든 장치의 메모리 완전 정리 (스레드 안전)"""
        with self._cleanup_lock:
            self.logger.info("전체 장치 메모리 정리 시작...")

            # 1. 모든 등록된 모델 정리
            active_models = self.model_registry.list_active_models()
            failed_models = []

            for model_name in active_models:
                success = self.cleanup_model(model_name, force=force)
                if not success:
                    failed_models.append(model_name)

            if failed_models:
                self.logger.warning(f"정리 실패 모델: {failed_models}")

            # 2. 죽은 참조 정리
            dead_count = self.model_registry.cleanup_dead_references()
            if dead_count > 0:
                self.logger.info(f"죽은 참조 {dead_count}개 정리됨")

            # 3. 모든 장치 강제 정리
            if force:
                cleanup_results = self.cleanup_manager.force_emergency_cleanup()
                self.logger.info(f"긴급 정리 결과: {cleanup_results}")
            else:
                # 개별 장치 정리
                devices = ["cpu"]
                if torch.cuda.is_available():
                    devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])

                for device in devices:
                    self.cleanup_manager.cleanup_device_memory(device)

            # 4. 추가 정리 콜백 실행
            self._execute_cleanup_callbacks()

            self.logger.info("전체 장치 메모리 정리 완료")

    def _execute_cleanup_callbacks(self):
        """정리 콜백 실행 (예외 안전)"""
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.warning(f"정리 콜백 실행 실패: {e}")

    def get_memory_stats(self, device: Optional[str] = None) -> Dict[str, MemoryStats]:
        """메모리 통계 조회 (스레드 안전)"""
        stats = {}

        if device is None:
            # 모든 장치 조회
            devices = ["cpu"]
            if torch.cuda.is_available():
                devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
        else:
            devices = [device]

        for dev in devices:
            try:
                if dev == "cpu":
                    stats[dev] = self._get_cpu_memory_stats()
                elif dev.startswith("cuda"):
                    device_id = int(dev.split(":")[-1])
                    stats[dev] = self._get_gpu_memory_stats(device_id)
            except Exception as e:
                self.logger.warning(f"장치 {dev} 메모리 통계 조회 실패: {e}")

        return stats

    def _get_cpu_memory_stats(self) -> MemoryStats:
        """CPU 메모리 통계"""
        memory = psutil.virtual_memory()

        total_gb = memory.total / (1024**3)
        used_gb = memory.used / (1024**3)
        free_gb = memory.available / (1024**3)
        utilization = memory.percent / 100

        # 위험 수준 판단
        if utilization < self.warning_threshold:
            level = MemoryLevel.SAFE
        elif utilization < self.critical_threshold:
            level = MemoryLevel.WARNING
        elif utilization < self.emergency_threshold:
            level = MemoryLevel.CRITICAL
        else:
            level = MemoryLevel.EMERGENCY

        return MemoryStats(
            device="cpu",
            total_gb=total_gb,
            allocated_gb=used_gb,
            reserved_gb=used_gb,
            free_gb=free_gb,
            utilization=utilization,
            level=level,
            timestamp=time.time()
        )

    def _get_gpu_memory_stats(self, device_id: int) -> MemoryStats:
        """GPU 메모리 통계"""
        with torch.cuda.device(device_id):
            total_bytes = torch.cuda.get_device_properties(device_id).total_memory
            allocated_bytes = torch.cuda.memory_allocated(device_id)
            reserved_bytes = torch.cuda.memory_reserved(device_id)

            total_gb = total_bytes / (1024**3)
            allocated_gb = allocated_bytes / (1024**3)
            reserved_gb = reserved_bytes / (1024**3)
            free_gb = total_gb - reserved_gb
            utilization = allocated_gb / total_gb

            # 위험 수준 판단
            if utilization < self.warning_threshold:
                level = MemoryLevel.SAFE
            elif utilization < self.critical_threshold:
                level = MemoryLevel.WARNING
            elif utilization < self.emergency_threshold:
                level = MemoryLevel.CRITICAL
            else:
                level = MemoryLevel.EMERGENCY

            return MemoryStats(
                device=f"cuda:{device_id}",
                total_gb=total_gb,
                allocated_gb=allocated_gb,
                reserved_gb=reserved_gb,
                free_gb=free_gb,
                utilization=utilization,
                level=level,
                timestamp=time.time()
            )

    def check_memory_safety(self, required_gb: float = 0, device: str = "auto") -> bool:
        """메모리 안전성 확인"""
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        stats = self.get_memory_stats(device)
        if device not in stats:
            return False

        stat = stats[device]

        # 요구 메모리 + 안전 마진
        needed_gb = required_gb + (stat.total_gb * 0.1)  # 10% 안전 마진

        return stat.free_gb >= needed_gb and stat.level != MemoryLevel.EMERGENCY

    def emergency_cleanup_if_needed(self) -> bool:
        """필요시 긴급 메모리 정리"""
        stats = self.get_memory_stats()

        emergency_needed = any(
            stat.level == MemoryLevel.EMERGENCY
            for stat in stats.values()
        )

        if emergency_needed:
            self.logger.warning("긴급 메모리 정리 실행...")
            self.cleanup_all_devices(force=True)
            return True

        return False

    @contextlib.contextmanager
    def memory_guard(self, model_name: str, required_gb: float = 0, device: str = "auto"):
        """메모리 보호 컨텍스트 매니저 (스레드 안전)"""
        # 진입 시 메모리 확인
        if not self.check_memory_safety(required_gb, device):
            self.emergency_cleanup_if_needed()

            if not self.check_memory_safety(required_gb, device):
                raise RuntimeError(f"메모리 부족: {required_gb}GB 필요, {device}")

        initial_stats = self.get_memory_stats(device)

        try:
            yield
        finally:
            # 종료 시 정리
            try:
                self.cleanup_model(model_name, force=True)
            except Exception as e:
                self.logger.warning(f"메모리 가드 종료 시 정리 실패: {e}")

            final_stats = self.get_memory_stats(device)

            # 메모리 누수 감지
            if device in initial_stats and device in final_stats:
                initial_used = initial_stats[device].allocated_gb
                final_used = final_stats[device].allocated_gb

                if final_used > initial_used + 0.5:  # 0.5GB 이상 증가시 경고
                    self.logger.warning(
                        f"잠재적 메모리 누수 감지: {final_used - initial_used:.2f}GB 증가"
                    )

    def start_monitoring(self, interval: float = 30.0):
        """메모리 모니터링 시작 (스레드 안전)"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_interval = interval
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self._monitor_thread.start()
        self.logger.info(f"메모리 모니터링 시작 (간격: {interval}초)")

    def stop_monitoring(self):
        """메모리 모니터링 중단"""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("메모리 모니터링 중단")

    def _monitoring_loop(self):
        """모니터링 루프 (예외 안전)"""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()

                for device, stat in stats.items():
                    if stat.level == MemoryLevel.WARNING:
                        self.logger.warning(
                            f"장치 {device} 메모리 사용량 주의: {stat.utilization:.1%}"
                        )
                    elif stat.level == MemoryLevel.CRITICAL:
                        self.logger.error(
                            f"장치 {device} 메모리 사용량 위험: {stat.utilization:.1%}"
                        )
                        # 자동 정리 시도
                        self.cleanup_manager.cleanup_device_memory(device)
                    elif stat.level == MemoryLevel.EMERGENCY:
                        self.logger.critical(
                            f"장치 {device} 메모리 사용량 응급: {stat.utilization:.1%} - 긴급 정리 실행"
                        )
                        self.emergency_cleanup_if_needed()

                time.sleep(self._monitor_interval)

            except Exception as e:
                self.logger.error(f"메모리 모니터링 오류: {e}")
                time.sleep(self._monitor_interval)

    def add_cleanup_callback(self, callback: Callable[[], None]):
        """정리 콜백 추가"""
        self._cleanup_callbacks.append(callback)

    def get_active_models(self) -> List[str]:
        """활성 모델 목록 반환"""
        return self.model_registry.list_active_models()

    def _emergency_shutdown(self):
        """긴급 종료 (atexit 콜백)"""
        try:
            self.stop_monitoring()
            self.cleanup_all_devices(force=True)
        except:
            pass  # 종료 시에는 예외 무시

    def __del__(self):
        """소멸자 - 모든 리소스 정리"""
        self._emergency_shutdown()


# 전역 리소스 매니저 인스턴스 (스레드 안전)
_global_resource_manager: Optional[ImprovedResourceManager] = None
_manager_lock = threading.RLock()


def get_resource_manager() -> ImprovedResourceManager:
    """전역 리소스 매니저 반환 (싱글톤, 스레드 안전)"""
    global _global_resource_manager

    with _manager_lock:
        if _global_resource_manager is None:
            _global_resource_manager = ImprovedResourceManager()
            _global_resource_manager.start_monitoring()  # 자동 모니터링 시작

    return _global_resource_manager


def cleanup_all_resources(force: bool = False):
    """전역 리소스 정리"""
    global _global_resource_manager

    with _manager_lock:
        if _global_resource_manager is not None:
            _global_resource_manager.stop_monitoring()
            _global_resource_manager.cleanup_all_devices(force=force)
            _global_resource_manager = None


# 사용 예시
if __name__ == "__main__":
    import time

    # 리소스 매니저 테스트
    manager = get_resource_manager()

    print("=== 개선된 메모리 관리 시스템 테스트 ===")

    # 메모리 통계 조회
    stats = manager.get_memory_stats()
    for device, stat in stats.items():
        print(f"{device}: {stat.allocated_gb:.2f}GB / {stat.total_gb:.2f}GB ({stat.utilization:.1%}) - {stat.level.value}")

    # 메모리 가드 사용 예시
    try:
        with manager.memory_guard("test_model", required_gb=1.0):
            print("메모리 보호 구간에서 모델 작업 수행...")
            # 실제 모델 작업 시뮬레이션
            import torch
            test_tensor = torch.randn(1000, 1000)
            manager.register_model("test_model", test_tensor, "cpu")
            time.sleep(1)
    except Exception as e:
        print(f"메모리 가드 오류: {e}")

    # 활성 모델 확인
    active_models = manager.get_active_models()
    print(f"활성 모델: {active_models}")

    # 정리
    cleanup_all_resources(force=True)
    print("✅ 개선된 메모리 관리 시스템 테스트 완료")