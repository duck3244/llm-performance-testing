"""
개선된 메모리 관리 시스템
Critical 문제 해결: GPU 메모리 완전 해제 로직 수정
"""
import gc
import time
import torch
import psutil
import threading
import weakref
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import contextlib


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


class ImprovedResourceManager:
    """개선된 리소스 관리자 - 완전한 메모리 해제 보장"""

    def __init__(self):
        self._active_models: Dict[str, weakref.ref] = {}
        self._device_locks: Dict[str, threading.Lock] = {}
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False
        self.logger = logging.getLogger(__name__)

        # 메모리 임계값 설정
        self.warning_threshold = 0.70
        self.critical_threshold = 0.85
        self.emergency_threshold = 0.95

        # 장치별 락 초기화
        self._init_device_locks()

    def _init_device_locks(self):
        """장치별 락 초기화"""
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_id = f"cuda:{i}"
                    self._device_locks[device_id] = threading.Lock()
        except Exception as e:
            self.logger.warning(f"CUDA 장치 초기화 실패: {e}")

        # CPU 락
        self._device_locks["cpu"] = threading.Lock()

    def register_model(self, model_name: str, model_ref: Any, device: str = "auto"):
        """모델 등록 및 추적"""
        with self._device_locks.get(device, threading.Lock()):
            # Weak reference로 등록하여 순환 참조 방지
            # 콜백에서 model_name을 캡처하기 위해 클로저 사용
            def cleanup_callback(ref):
                self._on_model_deleted(model_name)

            self._active_models[model_name] = {
                'ref': weakref.ref(model_ref, cleanup_callback),
                'device': device,
                'registered_at': time.time()
            }
            self.logger.info(f"모델 등록됨: {model_name} on {device}")

    def _on_model_deleted(self, model_name: str):
        """모델 삭제 시 자동 콜백"""
        if model_name in self._active_models:
            device = self._active_models[model_name]['device']
            del self._active_models[model_name]
            self.logger.info(f"모델 자동 정리됨: {model_name}")

            # 해당 장치 메모리 정리
            self._cleanup_device_memory(device)

    def cleanup_model(self, model_name: str) -> bool:
        """특정 모델 완전 정리"""
        if model_name not in self._active_models:
            return False

        model_info = self._active_models[model_name]
        device = model_info['device']
        model_ref = model_info['ref']()

        with self._device_locks.get(device, threading.Lock()):
            try:
                # 1. 모델을 CPU로 이동
                if model_ref is not None and hasattr(model_ref, 'cpu'):
                    self.logger.debug(f"모델 {model_name}을 CPU로 이동 중...")
                    model_ref.cpu()

                # 2. 모델 참조 제거
                if model_ref is not None:
                    del model_ref

                # 3. 등록에서 제거
                del self._active_models[model_name]

                # 4. 장치별 메모리 정리
                self._cleanup_device_memory(device)

                self.logger.info(f"모델 {model_name} 완전 정리 완료")
                return True

            except Exception as e:
                self.logger.error(f"모델 {model_name} 정리 실패: {e}")
                return False

    def _cleanup_device_memory(self, device: str):
        """장치별 메모리 완전 정리"""
        try:
            if device.startswith("cuda"):
                device_id = int(device.split(":")[-1]) if ":" in device else 0
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # IPC 메모리도 정리
                    torch.cuda.ipc_collect()

            elif device == "cpu":
                # CPU 메모리 정리
                gc.collect()

        except Exception as e:
            self.logger.warning(f"장치 {device} 메모리 정리 실패: {e}")

    def cleanup_all_devices(self):
        """모든 장치의 메모리 완전 정리"""
        self.logger.info("전체 장치 메모리 정리 시작...")

        # 1. 모든 등록된 모델 정리
        model_names = list(self._active_models.keys())
        for model_name in model_names:
            self.cleanup_model(model_name)

        # 2. 모든 CUDA 장치 정리
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.ipc_collect()
                        # 메모리 풀 재설정 (PyTorch 2.0+)
                        if hasattr(torch.cuda, 'memory_stats'):
                            torch.cuda.reset_peak_memory_stats(i)
                except Exception as e:
                    self.logger.warning(f"CUDA 장치 {i} 정리 실패: {e}")

        # 3. Python 가비지 컬렉션
        for _ in range(3):  # 여러 번 실행으로 순환 참조 완전 제거
            gc.collect()

        # 4. 추가 정리 콜백 실행
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.warning(f"정리 콜백 실행 실패: {e}")

        self.logger.info("전체 장치 메모리 정리 완료")

    def get_memory_stats(self, device: Optional[str] = None) -> Dict[str, MemoryStats]:
        """메모리 통계 조회"""
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
            reserved_gb=used_gb,  # CPU는 동일
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
            self.cleanup_all_devices()
            return True

        return False

    def start_monitoring(self, interval: float = 30.0):
        """메모리 모니터링 시작"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"메모리 모니터링 시작 (간격: {interval}초)")

    def stop_monitoring(self):
        """메모리 모니터링 중단"""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("메모리 모니터링 중단")

    def _monitoring_loop(self, interval: float):
        """모니터링 루프"""
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
                    elif stat.level == MemoryLevel.EMERGENCY:
                        self.logger.critical(
                            f"장치 {device} 메모리 사용량 응급: {stat.utilization:.1%} - 긴급 정리 실행"
                        )
                        self.emergency_cleanup_if_needed()

                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"메모리 모니터링 오류: {e}")
                time.sleep(interval)

    def add_cleanup_callback(self, callback: Callable[[], None]):
        """정리 콜백 추가"""
        self._cleanup_callbacks.append(callback)

    @contextlib.contextmanager
    def memory_guard(self, model_name: str, required_gb: float = 0, device: str = "auto"):
        """메모리 보호 컨텍스트 매니저"""
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
            if model_name in self._active_models:
                self.cleanup_model(model_name)

            final_stats = self.get_memory_stats(device)

            # 메모리 누수 감지
            if device in initial_stats and device in final_stats:
                initial_used = initial_stats[device].allocated_gb
                final_used = final_stats[device].allocated_gb

                if final_used > initial_used + 0.5:  # 0.5GB 이상 증가시 경고
                    self.logger.warning(
                        f"잠재적 메모리 누수 감지: {final_used - initial_used:.2f}GB 증가"
                    )

    def get_active_models(self) -> List[str]:
        """활성 모델 목록 반환"""
        active = []
        for name, info in self._active_models.items():
            if info['ref']() is not None:  # 아직 살아있는 모델만
                active.append(name)
        return active

    def __del__(self):
        """소멸자 - 모든 리소스 정리"""
        try:
            self.stop_monitoring()
            self.cleanup_all_devices()
        except:
            pass  # 소멸자에서는 예외 무시


# 전역 리소스 매니저 인스턴스
_global_resource_manager: Optional[ImprovedResourceManager] = None
_manager_lock = threading.Lock()


def get_resource_manager() -> ImprovedResourceManager:
    """전역 리소스 매니저 반환 (싱글톤)"""
    global _global_resource_manager

    with _manager_lock:
        if _global_resource_manager is None:
            _global_resource_manager = ImprovedResourceManager()
            _global_resource_manager.start_monitoring()  # 자동 모니터링 시작

    return _global_resource_manager


def cleanup_all_resources():
    """전역 리소스 정리"""
    global _global_resource_manager

    with _manager_lock:
        if _global_resource_manager is not None:
            _global_resource_manager.stop_monitoring()
            _global_resource_manager.cleanup_all_devices()
            _global_resource_manager = None


# 프로그램 종료시 자동 정리
import atexit
atexit.register(cleanup_all_resources)


# 사용 예시
if __name__ == "__main__":
    import time

    # 리소스 매니저 테스트
    manager = get_resource_manager()

    print("=== 메모리 관리 시스템 테스트 ===")

    # 메모리 통계 조회
    stats = manager.get_memory_stats()
    for device, stat in stats.items():
        print(f"{device}: {stat.allocated_gb:.2f}GB / {stat.total_gb:.2f}GB ({stat.utilization:.1%}) - {stat.level.value}")

    # 메모리 가드 사용 예시
    try:
        with manager.memory_guard("test_model", required_gb=1.0):
            print("메모리 보호 구간에서 모델 작업 수행...")
            time.sleep(1)
    except Exception as e:
        print(f"메모리 가드 오류: {e}")

    # 정리
    cleanup_all_resources()
    print("✅ 메모리 관리 시스템 테스트 완료")