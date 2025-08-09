"""
개선된 비동기 처리 관리자
Critical 문제 해결: 스레드 안전성 및 이벤트 루프 관리 개선
"""
import asyncio
import threading
import functools
import logging
import time
import weakref
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
import contextlib

T = TypeVar('T')


class ExecutorType(Enum):
    """실행자 타입"""
    THREAD = "thread"
    PROCESS = "process"
    ASYNCIO = "asyncio"


@dataclass
class TaskInfo:
    """태스크 정보"""
    task_id: str
    name: str
    created_at: float
    executor_type: ExecutorType
    status: str = "pending"
    result: Any = None
    error: Optional[Exception] = None


class SafeAsyncManager:
    """스레드 안전한 비동기 관리자"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

        # 실행자들
        self._thread_executor: Optional[ThreadPoolExecutor] = None
        self._process_executor: Optional[ProcessPoolExecutor] = None

        # 이벤트 루프 관리
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_started = threading.Event()

        # 태스크 추적
        self._active_tasks: Dict[str, TaskInfo] = {}
        self._task_counter = 0
        self._lock = threading.RLock()

        # 종료 상태
        self._shutdown = False

        # 이벤트 루프 시작
        self._start_event_loop()

    def _start_event_loop(self):
        """전용 스레드에서 이벤트 루프 시작"""

        def run_loop():
            """이벤트 루프 실행"""
            try:
                # 새 이벤트 루프 생성
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

                self.logger.info("전용 이벤트 루프 시작")
                self._loop_started.set()

                # 루프 실행
                self._loop.run_forever()

            except Exception as e:
                self.logger.error(f"이벤트 루프 실행 오류: {e}")
            finally:
                if self._loop:
                    self._loop.close()
                self.logger.info("이벤트 루프 종료")

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

        # 루프가 시작될 때까지 대기
        self._loop_started.wait(timeout=10.0)

        if not self._loop_started.is_set():
            raise RuntimeError("이벤트 루프 시작 실패")

    def _get_thread_executor(self) -> ThreadPoolExecutor:
        """스레드 풀 실행자 반환"""
        if self._thread_executor is None:
            self._thread_executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="SafeAsync"
            )
        return self._thread_executor

    def _get_process_executor(self) -> ProcessPoolExecutor:
        """프로세스 풀 실행자 반환"""
        if self._process_executor is None:
            self._process_executor = ProcessPoolExecutor(
                max_workers=min(self.max_workers, 4)  # 프로세스는 제한적으로
            )
        return self._process_executor

    def _generate_task_id(self) -> str:
        """태스크 ID 생성"""
        with self._lock:
            self._task_counter += 1
            return f"task_{self._task_counter}_{int(time.time())}"

    def run_async(self, coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
        """코루틴을 안전하게 실행"""
        if self._shutdown:
            raise RuntimeError("AsyncManager가 종료되었습니다")

        if not self._loop or self._loop.is_closed():
            raise RuntimeError("이벤트 루프가 사용할 수 없습니다")

        # 현재 스레드가 이벤트 루프 스레드인지 확인
        if threading.current_thread() == self._loop_thread:
            # 같은 스레드에서는 직접 실행
            return self._loop.run_until_complete(coro)

        # 다른 스레드에서는 thread-safe하게 실행
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        try:
            return future.result(timeout=timeout)
        except asyncio.TimeoutError:
            future.cancel()
            raise TimeoutError(f"태스크가 {timeout}초 내에 완료되지 않았습니다")

    def submit_coro(self, coro: Coroutine[Any, Any, T],
                    name: Optional[str] = None) -> str:
        """코루틴을 비동기로 제출"""
        task_id = self._generate_task_id()
        task_name = name or f"coro_{task_id}"

        with self._lock:
            self._active_tasks[task_id] = TaskInfo(
                task_id=task_id,
                name=task_name,
                created_at=time.time(),
                executor_type=ExecutorType.ASYNCIO
            )

        # 태스크 생성 및 실행
        def create_task():
            try:
                task = self._loop.create_task(coro)
                task.add_done_callback(
                    lambda t: self._on_task_done(task_id, t)
                )
                return task
            except Exception as e:
                self._on_task_error(task_id, e)

        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(create_task(), self._loop)

        return task_id

    def submit_func(self, func: Callable[..., T], *args,
                    executor_type: ExecutorType = ExecutorType.THREAD,
                    name: Optional[str] = None, **kwargs) -> str:
        """함수를 지정된 실행자에서 실행"""
        task_id = self._generate_task_id()
        task_name = name or f"func_{func.__name__}_{task_id}"

        with self._lock:
            self._active_tasks[task_id] = TaskInfo(
                task_id=task_id,
                name=task_name,
                created_at=time.time(),
                executor_type=executor_type
            )

        try:
            if executor_type == ExecutorType.THREAD:
                executor = self._get_thread_executor()
                future = executor.submit(func, *args, **kwargs)
            elif executor_type == ExecutorType.PROCESS:
                executor = self._get_process_executor()
                future = executor.submit(func, *args, **kwargs)
            else:
                raise ValueError(f"지원하지 않는 실행자 타입: {executor_type}")

            # 완료 콜백 추가
            future.add_done_callback(
                lambda f: self._on_future_done(task_id, f)
            )

        except Exception as e:
            self._on_task_error(task_id, e)

        return task_id

    def _on_task_done(self, task_id: str, task: asyncio.Task):
        """비동기 태스크 완료 처리"""
        with self._lock:
            if task_id in self._active_tasks:
                task_info = self._active_tasks[task_id]

                try:
                    if task.cancelled():
                        task_info.status = "cancelled"
                    elif task.exception():
                        task_info.status = "failed"
                        task_info.error = task.exception()
                    else:
                        task_info.status = "completed"
                        task_info.result = task.result()
                except Exception as e:
                    task_info.status = "failed"
                    task_info.error = e

                self.logger.debug(f"태스크 {task_id} 완료: {task_info.status}")

    def _on_future_done(self, task_id: str, future):
        """Future 완료 처리"""
        with self._lock:
            if task_id in self._active_tasks:
                task_info = self._active_tasks[task_id]

                try:
                    if future.cancelled():
                        task_info.status = "cancelled"
                    elif future.exception():
                        task_info.status = "failed"
                        task_info.error = future.exception()
                    else:
                        task_info.status = "completed"
                        task_info.result = future.result()
                except Exception as e:
                    task_info.status = "failed"
                    task_info.error = e

                self.logger.debug(f"Future {task_id} 완료: {task_info.status}")

    def _on_task_error(self, task_id: str, error: Exception):
        """태스크 오류 처리"""
        with self._lock:
            if task_id in self._active_tasks:
                task_info = self._active_tasks[task_id]
                task_info.status = "failed"
                task_info.error = error

                self.logger.error(f"태스크 {task_id} 실행 오류: {error}")

    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """태스크 결과 대기 및 반환"""
        start_time = time.time()

        while True:
            with self._lock:
                if task_id not in self._active_tasks:
                    raise ValueError(f"태스크 {task_id}를 찾을 수 없습니다")

                task_info = self._active_tasks[task_id]

                if task_info.status == "completed":
                    return task_info.result
                elif task_info.status == "failed":
                    raise task_info.error or RuntimeError(f"태스크 {task_id} 실행 실패")
                elif task_info.status == "cancelled":
                    raise asyncio.CancelledError(f"태스크 {task_id}가 취소되었습니다")

            # 타임아웃 확인
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"태스크 {task_id}가 {timeout}초 내에 완료되지 않았습니다")

            time.sleep(0.01)  # 짧은 대기

    def cancel_task(self, task_id: str) -> bool:
        """태스크 취소"""
        with self._lock:
            if task_id not in self._active_tasks:
                return False

            task_info = self._active_tasks[task_id]

            if task_info.status in ["completed", "failed", "cancelled"]:
                return False

            # 취소 시도 (실제 구현은 복잡할 수 있음)
            task_info.status = "cancelled"
            self.logger.info(f"태스크 {task_id} 취소 요청")

            return True

    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """태스크 상태 조회"""
        with self._lock:
            return self._active_tasks.get(task_id)

    def list_active_tasks(self) -> List[TaskInfo]:
        """활성 태스크 목록"""
        with self._lock:
            return [
                task_info for task_info in self._active_tasks.values()
                if task_info.status == "pending"
            ]

    def cleanup_completed_tasks(self, max_age: float = 3600.0):
        """완료된 태스크 정리 (기본 1시간)"""
        current_time = time.time()
        to_remove = []

        with self._lock:
            for task_id, task_info in self._active_tasks.items():
                if (task_info.status in ["completed", "failed", "cancelled"] and
                        current_time - task_info.created_at > max_age):
                    to_remove.append(task_id)

            for task_id in to_remove:
                del self._active_tasks[task_id]

        if to_remove:
            self.logger.info(f"정리된 완료 태스크: {len(to_remove)}개")

    @contextlib.asynccontextmanager
    async def managed_task(self, coro: Coroutine[Any, Any, T],
                           name: Optional[str] = None):
        """관리되는 태스크 컨텍스트"""
        task_id = None
        try:
            task_id = self.submit_coro(coro, name)
            result = self.get_task_result(task_id)
            yield result
        except Exception as e:
            if task_id:
                self.cancel_task(task_id)
            raise
        finally:
            if task_id and task_id in self._active_tasks:
                # 태스크 정보는 나중에 정리되도록 유지
                pass

    def run_parallel(self, coroutines: List[Coroutine],
                     timeout: Optional[float] = None) -> List[Any]:
        """여러 코루틴을 병렬로 실행"""
        if not coroutines:
            return []

        async def gather_coros():
            try:
                return await asyncio.gather(*coroutines, return_exceptions=True)
            except Exception as e:
                self.logger.error(f"병렬 실행 오류: {e}")
                raise

        return self.run_async(gather_coros(), timeout)

    def run_with_semaphore(self, coroutines: List[Coroutine],
                           max_concurrent: int = 5) -> List[Any]:
        """세마포어를 사용한 제한된 병렬 실행"""

        async def limited_gather():
            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_with_sem(coro):
                async with semaphore:
                    return await coro

            tasks = [run_with_sem(coro) for coro in coroutines]
            return await asyncio.gather(*tasks, return_exceptions=True)

        return self.run_async(limited_gather())

    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """매니저 종료"""
        if self._shutdown:
            return

        self.logger.info("AsyncManager 종료 시작...")
        self._shutdown = True

        try:
            # 활성 태스크들 취소
            with self._lock:
                for task_id in list(self._active_tasks.keys()):
                    self.cancel_task(task_id)

            # 실행자들 종료
            if self._thread_executor:
                self._thread_executor.shutdown(wait=wait, timeout=timeout)
                self._thread_executor = None

            if self._process_executor:
                self._process_executor.shutdown(wait=wait, timeout=timeout)
                self._process_executor = None

            # 이벤트 루프 종료
            if self._loop and not self._loop.is_closed():
                if wait:
                    # 안전하게 종료
                    self._loop.call_soon_threadsafe(self._loop.stop)
                    if self._loop_thread and self._loop_thread.is_alive():
                        self._loop_thread.join(timeout=timeout)
                else:
                    # 즉시 종료
                    self._loop.call_soon_threadsafe(self._loop.stop)

            self.logger.info("AsyncManager 종료 완료")

        except Exception as e:
            self.logger.error(f"AsyncManager 종료 중 오류: {e}")

    def __del__(self):
        """소멸자"""
        try:
            self.shutdown(wait=False)
        except:
            pass


# 전역 비동기 매니저
_global_async_manager: Optional[SafeAsyncManager] = None
_async_manager_lock = threading.Lock()


def get_async_manager() -> SafeAsyncManager:
    """전역 비동기 매니저 반환 (싱글톤)"""
    global _global_async_manager

    with _async_manager_lock:
        if _global_async_manager is None:
            _global_async_manager = SafeAsyncManager()

    return _global_async_manager


def cleanup_async_manager():
    """전역 비동기 매니저 정리"""
    global _global_async_manager

    with _async_manager_lock:
        if _global_async_manager is not None:
            _global_async_manager.shutdown()
            _global_async_manager = None


# 편의 함수들
def run_async_safe(coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
    """안전한 비동기 실행"""
    manager = get_async_manager()
    return manager.run_async(coro, timeout)


def submit_async_task(coro: Coroutine, name: Optional[str] = None) -> str:
    """비동기 태스크 제출"""
    manager = get_async_manager()
    return manager.submit_coro(coro, name)


def run_parallel_safe(coroutines: List[Coroutine], timeout: Optional[float] = None) -> List[Any]:
    """안전한 병렬 실행"""
    manager = get_async_manager()
    return manager.run_parallel(coroutines, timeout)


# 데코레이터
def async_safe(timeout: Optional[float] = None):
    """비동기 안전 실행 데코레이터"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                coro = func(*args, **kwargs)
                return run_async_safe(coro, timeout)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


# 프로그램 종료시 자동 정리
import atexit

atexit.register(cleanup_async_manager)

# 사용 예시
if __name__ == "__main__":
    import time


    async def example_coro(delay: float, value: int):
        """예시 코루틴"""
        await asyncio.sleep(delay)
        return value * 2


    def example_func(delay: float, value: int):
        """예시 함수"""
        time.sleep(delay)
        return value * 3


    # 비동기 매니저 테스트
    manager = get_async_manager()

    print("=== 비동기 관리 시스템 테스트 ===")

    # 1. 단일 코루틴 실행
    result = manager.run_async(example_coro(0.1, 5))
    print(f"단일 코루틴 결과: {result}")

    # 2. 병렬 실행
    coroutines = [example_coro(0.1, i) for i in range(3)]
    results = manager.run_parallel(coroutines)
    print(f"병렬 실행 결과: {results}")

    # 3. 태스크 제출 및 결과 대기
    task_id = manager.submit_func(example_func, 0.1, 10, ExecutorType.THREAD)
    task_result = manager.get_task_result(task_id, timeout=5.0)
    print(f"태스크 결과: {task_result}")

    # 4. 활성 태스크 조회
    active_tasks = manager.list_active_tasks()
    print(f"활성 태스크 수: {len(active_tasks)}")

    # 5. 제한된 병렬 실행
    many_coros = [example_coro(0.05, i) for i in range(10)]
    limited_results = manager.run_with_semaphore(many_coros, max_concurrent=3)
    print(f"제한된 병렬 실행 결과: {len(limited_results)}개")

    # 정리
    manager.cleanup_completed_tasks()
    cleanup_async_manager()

    print("✅ 비동기 관리 시스템 테스트 완료")