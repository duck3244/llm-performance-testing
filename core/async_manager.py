"""
개선된 비동기 처리 관리자 - 동시성 문제 완전 해결
Critical 문제 해결: 스레드 안전성 강화, 데드락 방지, 이벤트 루프 안정성
"""
import asyncio
import threading
import functools
import logging
import time
import weakref
import uuid
import queue
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Union, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
import contextlib
from collections import defaultdict
import atexit


T = TypeVar('T')


class ExecutorType(Enum):
    """실행자 타입"""
    THREAD = "thread"
    PROCESS = "process"
    ASYNCIO = "asyncio"


class TaskState(Enum):
    """태스크 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskInfo:
    """태스크 정보"""
    task_id: str
    name: str
    created_at: float
    executor_type: ExecutorType
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Optional[Exception] = None
    timeout: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class ThreadSafeTaskRegistry:
    """스레드 안전한 태스크 레지스트리"""

    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._task_locks: Dict[str, threading.RLock] = {}
        self._registry_lock = threading.RLock()
        self._cleanup_queue = queue.Queue()

        # 백그라운드 정리 스레드
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="TaskCleanupWorker"
        )
        self._cleanup_running = True
        self._cleanup_thread.start()

    def register_task(self, task_info: TaskInfo) -> bool:
        """태스크 등록"""
        with self._registry_lock:
            if task_info.task_id in self._tasks:
                return False

            self._task_locks[task_info.task_id] = threading.RLock()
            self._tasks[task_info.task_id] = task_info
            return True

    def get_task_lock(self, task_id: str) -> Optional[threading.RLock]:
        """태스크별 락 반환"""
        with self._registry_lock:
            return self._task_locks.get(task_id)

    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """태스크 정보 조회"""
        with self._registry_lock:
            return self._tasks.get(task_id)

    def update_task_state(self, task_id: str, state: TaskState,
                         result: Any = None, error: Exception = None):
        """태스크 상태 업데이트"""
        task_lock = self.get_task_lock(task_id)
        if not task_lock:
            return False

        with task_lock:
            task_info = self._tasks.get(task_id)
            if not task_info:
                return False

            task_info.state = state
            task_info.result = result
            task_info.error = error

            if state == TaskState.RUNNING and task_info.started_at is None:
                task_info.started_at = time.time()
            elif state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED, TaskState.TIMEOUT]:
                task_info.completed_at = time.time()
                # 완료된 태스크는 정리 대상으로 추가
                self._schedule_cleanup(task_id)

            return True

    def list_active_tasks(self) -> List[TaskInfo]:
        """활성 태스크 목록"""
        with self._registry_lock:
            return [
                task_info for task_info in self._tasks.values()
                if task_info.state in [TaskState.PENDING, TaskState.RUNNING]
            ]

    def list_completed_tasks(self, max_age_seconds: float = 3600) -> List[TaskInfo]:
        """완료된 태스크 목록"""
        current_time = time.time()
        with self._registry_lock:
            return [
                task_info for task_info in self._tasks.values()
                if (task_info.state in [TaskState.COMPLETED, TaskState.FAILED,
                                      TaskState.CANCELLED, TaskState.TIMEOUT] and
                    task_info.completed_at and
                    current_time - task_info.completed_at <= max_age_seconds)
            ]

    def _schedule_cleanup(self, task_id: str, delay: float = 300):
        """태스크 정리 예약 (5분 후)"""
        cleanup_time = time.time() + delay
        try:
            self._cleanup_queue.put((cleanup_time, task_id), timeout=1.0)
        except queue.Full:
            pass  # 큐가 가득 찬 경우 무시

    def _cleanup_worker(self):
        """백그라운드 정리 작업자"""
        logger = logging.getLogger(__name__)

        while self._cleanup_running:
            try:
                # 1초 타임아웃으로 정리 작업 대기
                cleanup_time, task_id = self._cleanup_queue.get(timeout=1.0)

                # 정리 시간까지 대기
                wait_time = cleanup_time - time.time()
                if wait_time > 0:
                    time.sleep(min(wait_time, 60))  # 최대 1분 대기

                # 태스크 정리
                with self._registry_lock:
                    if task_id in self._tasks:
                        task_info = self._tasks[task_id]
                        # 완료된 지 5분 이상 경과한 태스크만 정리
                        if (task_info.completed_at and
                            time.time() - task_info.completed_at >= 300):
                            del self._tasks[task_id]
                            del self._task_locks[task_id]
                            logger.debug(f"태스크 {task_id} 자동 정리됨")

            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"태스크 정리 중 오류: {e}")

    def cleanup_completed_tasks(self, max_age_seconds: float = 3600) -> int:
        """완료된 태스크 수동 정리"""
        current_time = time.time()
        cleaned_count = 0

        with self._registry_lock:
            to_remove = []
            for task_id, task_info in self._tasks.items():
                if (task_info.state in [TaskState.COMPLETED, TaskState.FAILED,
                                      TaskState.CANCELLED, TaskState.TIMEOUT] and
                    task_info.completed_at and
                    current_time - task_info.completed_at > max_age_seconds):
                    to_remove.append(task_id)

            for task_id in to_remove:
                del self._tasks[task_id]
                del self._task_locks[task_id]
                cleaned_count += 1

        return cleaned_count

    def shutdown(self):
        """정리 작업자 종료"""
        self._cleanup_running = False
        try:
            self._cleanup_thread.join(timeout=2.0)
        except:
            pass


class SafeEventLoopManager:
    """안전한 이벤트 루프 매니저"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_started = threading.Event()
        self._loop_stopping = threading.Event()
        self._shutdown = False

        # 이벤트 루프 통신을 위한 스레드 안전 큐
        self._command_queue = queue.Queue()
        self._result_futures: Dict[str, threading.Event] = {}
        self._results: Dict[str, Any] = {}

    def start_loop(self) -> bool:
        """전용 스레드에서 이벤트 루프 시작"""
        if self._loop_thread and self._loop_thread.is_alive():
            return True

        self._loop_started.clear()
        self._loop_stopping.clear()
        self._shutdown = False

        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="SafeEventLoop"
        )
        self._loop_thread.start()

        # 루프가 시작될 때까지 대기 (최대 10초)
        started = self._loop_started.wait(timeout=10.0)
        if not started:
            self.logger.error("이벤트 루프 시작 실패")
            return False

        self.logger.info("안전한 이벤트 루프 시작됨")
        return True

    def _run_event_loop(self):
        """이벤트 루프 실행"""
        try:
            # 새 이벤트 루프 생성
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # 명령 처리 태스크 시작
            self._loop.create_task(self._command_processor())

            self.logger.debug("이벤트 루프 준비 완료")
            self._loop_started.set()

            # 루프 실행
            self._loop.run_forever()

        except Exception as e:
            self.logger.error(f"이벤트 루프 실행 오류: {e}")
        finally:
            if self._loop:
                try:
                    # 남은 태스크들 정리
                    pending_tasks = [task for task in asyncio.all_tasks(self._loop)
                                   if not task.done()]
                    if pending_tasks:
                        self.logger.debug(f"남은 태스크 {len(pending_tasks)}개 정리 중...")
                        for task in pending_tasks:
                            task.cancel()

                    self._loop.close()
                except Exception as e:
                    self.logger.warning(f"이벤트 루프 정리 중 오류: {e}")

            self._loop_stopping.set()
            self.logger.debug("이벤트 루프 종료됨")

    async def _command_processor(self):
        """명령 처리기 (이벤트 루프 내에서 실행)"""
        while not self._shutdown:
            try:
                # 논블로킹으로 명령 확인
                await asyncio.sleep(0.01)

                try:
                    command_id, command_type, coro_or_func, timeout_val = self._command_queue.get_nowait()
                except queue.Empty:
                    continue

                try:
                    if command_type == "coro":
                        # 코루틴 실행
                        if timeout_val:
                            result = await asyncio.wait_for(coro_or_func, timeout=timeout_val)
                        else:
                            result = await coro_or_func
                    elif command_type == "task":
                        # 태스크 생성
                        task = self._loop.create_task(coro_or_func)
                        result = task
                    else:
                        result = Exception(f"알 수 없는 명령 타입: {command_type}")

                except Exception as e:
                    result = e

                # 결과 저장 및 이벤트 설정
                self._results[command_id] = result
                if command_id in self._result_futures:
                    self._result_futures[command_id].set()

            except Exception as e:
                self.logger.error(f"명령 처리 중 오류: {e}")

    def run_coroutine(self, coro: Coroutine, timeout: Optional[float] = None) -> Any:
        """코루틴을 안전하게 실행"""
        if self._shutdown:
            raise RuntimeError("이벤트 루프가 종료되었습니다")

        if not self._loop or self._loop.is_closed():
            raise RuntimeError("이벤트 루프가 사용할 수 없습니다")

        # 현재 스레드가 이벤트 루프 스레드인지 확인
        if threading.current_thread() == self._loop_thread:
            # 같은 스레드에서는 직접 실행 (데드락 방지)
            if timeout:
                return asyncio.get_event_loop().run_until_complete(
                    asyncio.wait_for(coro, timeout=timeout)
                )
            else:
                return asyncio.get_event_loop().run_until_complete(coro)

        # 다른 스레드에서는 명령 큐를 통해 실행
        command_id = str(uuid.uuid4())
        result_event = threading.Event()
        self._result_futures[command_id] = result_event

        try:
            # 명령 전송
            self._command_queue.put((command_id, "coro", coro, timeout), timeout=5.0)

            # 결과 대기
            wait_timeout = (timeout + 5) if timeout else 300  # 기본 5분
            if not result_event.wait(timeout=wait_timeout):
                raise TimeoutError(f"코루틴 실행이 {wait_timeout}초 내에 완료되지 않았습니다")

            # 결과 반환
            result = self._results.pop(command_id, None)
            if isinstance(result, Exception):
                raise result

            return result

        finally:
            # 정리
            self._result_futures.pop(command_id, None)
            self._results.pop(command_id, None)

    def create_task(self, coro: Coroutine) -> Any:
        """태스크 생성"""
        if self._shutdown:
            raise RuntimeError("이벤트 루프가 종료되었습니다")

        command_id = str(uuid.uuid4())
        result_event = threading.Event()
        self._result_futures[command_id] = result_event

        try:
            # 명령 전송
            self._command_queue.put((command_id, "task", coro, None), timeout=5.0)

            # 결과 대기 (태스크 생성은 빨라야 함)
            if not result_event.wait(timeout=10.0):
                raise TimeoutError("태스크 생성이 10초 내에 완료되지 않았습니다")

            # 결과 반환
            result = self._results.pop(command_id, None)
            if isinstance(result, Exception):
                raise result

            return result

        finally:
            # 정리
            self._result_futures.pop(command_id, None)
            self._results.pop(command_id, None)

    def stop_loop(self, timeout: float = 10.0):
        """이벤트 루프 안전 종료"""
        if self._shutdown:
            return

        self.logger.info("이벤트 루프 종료 시작...")
        self._shutdown = True

        if self._loop and not self._loop.is_closed():
            # 루프 정지 명령
            self._loop.call_soon_threadsafe(self._loop.stop)

        # 루프 스레드 종료 대기
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=timeout)

        # 강제 종료 확인
        if self._loop_thread and self._loop_thread.is_alive():
            self.logger.warning("이벤트 루프 스레드가 정상 종료되지 않음")
        else:
            self.logger.info("이벤트 루프 종료 완료")


class SafeAsyncManager:
    """스레드 안전한 비동기 관리자 - 동시성 문제 완전 해결"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

        # 컴포넌트 초기화
        self.task_registry = ThreadSafeTaskRegistry()
        self.event_loop_manager = SafeEventLoopManager()

        # 실행자들 (지연 초기화)
        self._thread_executor: Optional[ThreadPoolExecutor] = None
        self._process_executor: Optional[ProcessPoolExecutor] = None
        self._executor_lock = threading.RLock()

        # 태스크 관리
        self._task_counter = 0
        self._counter_lock = threading.Lock()

        # 종료 상태
        self._shutdown = False
        self._shutdown_lock = threading.RLock()

        # 이벤트 루프 시작
        if not self.event_loop_manager.start_loop():
            raise RuntimeError("이벤트 루프 시작 실패")

        # 프로그램 종료 시 자동 정리
        atexit.register(self._emergency_shutdown)

    def _get_thread_executor(self) -> ThreadPoolExecutor:
        """스레드 풀 실행자 반환 (지연 초기화)"""
        with self._executor_lock:
            if self._thread_executor is None:
                self._thread_executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="SafeAsync"
                )
            return self._thread_executor

    def _get_process_executor(self) -> ProcessPoolExecutor:
        """프로세스 풀 실행자 반환 (지연 초기화)"""
        with self._executor_lock:
            if self._process_executor is None:
                self._process_executor = ProcessPoolExecutor(
                    max_workers=min(self.max_workers, 4)  # 프로세스는 제한적으로
                )
            return self._process_executor

    def _generate_task_id(self) -> str:
        """고유한 태스크 ID 생성"""
        with self._counter_lock:
            self._task_counter += 1
            return f"task_{self._task_counter}_{int(time.time() * 1000000)}"

    def run_async(self, coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
        """코루틴을 안전하게 실행"""
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("AsyncManager가 종료되었습니다")

        try:
            return self.event_loop_manager.run_coroutine(coro, timeout)
        except Exception as e:
            self.logger.error(f"코루틴 실행 실패: {e}")
            raise

    def submit_coro(self, coro: Coroutine[Any, Any, T],
                    name: Optional[str] = None,
                    timeout: Optional[float] = None) -> str:
        """코루틴을 비동기로 제출"""
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("AsyncManager가 종료되었습니다")

        task_id = self._generate_task_id()
        task_name = name or f"coro_{task_id}"

        # 태스크 정보 생성 및 등록
        task_info = TaskInfo(
            task_id=task_id,
            name=task_name,
            created_at=time.time(),
            executor_type=ExecutorType.ASYNCIO,
            timeout=timeout
        )

        if not self.task_registry.register_task(task_info):
            raise RuntimeError(f"태스크 {task_id} 등록 실패")

        try:
            # 태스크 생성 및 콜백 설정
            async def wrapped_coro():
                try:
                    self.task_registry.update_task_state(task_id, TaskState.RUNNING)

                    if timeout:
                        result = await asyncio.wait_for(coro, timeout=timeout)
                    else:
                        result = await coro

                    self.task_registry.update_task_state(
                        task_id, TaskState.COMPLETED, result=result
                    )
                    return result

                except asyncio.TimeoutError:
                    self.task_registry.update_task_state(task_id, TaskState.TIMEOUT)
                    raise
                except asyncio.CancelledError:
                    self.task_registry.update_task_state(task_id, TaskState.CANCELLED)
                    raise
                except Exception as e:
                    self.task_registry.update_task_state(
                        task_id, TaskState.FAILED, error=e
                    )
                    raise

            # 태스크 생성
            task = self.event_loop_manager.create_task(wrapped_coro())
            self.logger.debug(f"태스크 {task_id} 생성됨")

            return task_id

        except Exception as e:
            self.task_registry.update_task_state(task_id, TaskState.FAILED, error=e)
            self.logger.error(f"태스크 {task_id} 생성 실패: {e}")
            raise

    def submit_func(self, func: Callable[..., T], *args,
                    executor_type: ExecutorType = ExecutorType.THREAD,
                    name: Optional[str] = None,
                    timeout: Optional[float] = None,
                    **kwargs) -> str:
        """함수를 지정된 실행자에서 실행"""
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("AsyncManager가 종료되었습니다")

        task_id = self._generate_task_id()
        task_name = name or f"func_{func.__name__}_{task_id}"

        # 태스크 정보 생성 및 등록
        task_info = TaskInfo(
            task_id=task_id,
            name=task_name,
            created_at=time.time(),
            executor_type=executor_type,
            timeout=timeout
        )

        if not self.task_registry.register_task(task_info):
            raise RuntimeError(f"태스크 {task_id} 등록 실패")

        try:
            # 실행자 선택 및 실행
            if executor_type == ExecutorType.THREAD:
                executor = self._get_thread_executor()
            elif executor_type == ExecutorType.PROCESS:
                executor = self._get_process_executor()
            else:
                raise ValueError(f"지원하지 않는 실행자 타입: {executor_type}")

            # 래핑된 함수로 상태 추적
            def wrapped_func(*args, **kwargs):
                try:
                    self.task_registry.update_task_state(task_id, TaskState.RUNNING)

                    # 타임아웃 처리
                    if timeout:
                        import signal

                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"함수 실행이 {timeout}초를 초과했습니다")

                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(timeout))

                        try:
                            result = func(*args, **kwargs)
                        finally:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old_handler)
                    else:
                        result = func(*args, **kwargs)

                    self.task_registry.update_task_state(
                        task_id, TaskState.COMPLETED, result=result
                    )
                    return result

                except TimeoutError:
                    self.task_registry.update_task_state(task_id, TaskState.TIMEOUT)
                    raise
                except Exception as e:
                    self.task_registry.update_task_state(
                        task_id, TaskState.FAILED, error=e
                    )
                    raise

            # Future 제출
            future = executor.submit(wrapped_func, *args, **kwargs)
            self.logger.debug(f"함수 태스크 {task_id} 제출됨")

            return task_id

        except Exception as e:
            self.task_registry.update_task_state(task_id, TaskState.FAILED, error=e)
            self.logger.error(f"함수 태스크 {task_id} 제출 실패: {e}")
            raise

    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """태스크 결과 대기 및 반환"""
        start_time = time.time()
        check_interval = 0.1  # 100ms 간격으로 확인

        while True:
            task_info = self.task_registry.get_task_info(task_id)
            if not task_info:
                raise ValueError(f"태스크 {task_id}를 찾을 수 없습니다")

            if task_info.state == TaskState.COMPLETED:
                return task_info.result
            elif task_info.state == TaskState.FAILED:
                raise task_info.error or RuntimeError(f"태스크 {task_id} 실행 실패")
            elif task_info.state == TaskState.CANCELLED:
                raise asyncio.CancelledError(f"태스크 {task_id}가 취소되었습니다")
            elif task_info.state == TaskState.TIMEOUT:
                raise TimeoutError(f"태스크 {task_id}가 시간 초과되었습니다")

            # 타임아웃 확인
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"태스크 {task_id}가 {timeout}초 내에 완료되지 않았습니다")

            time.sleep(check_interval)

    def cancel_task(self, task_id: str) -> bool:
        """태스크 취소"""
        task_lock = self.task_registry.get_task_lock(task_id)
        if not task_lock:
            return False

        with task_lock:
            task_info = self.task_registry.get_task_info(task_id)
            if not task_info:
                return False

            if task_info.state in [TaskState.COMPLETED, TaskState.FAILED,
                                 TaskState.CANCELLED, TaskState.TIMEOUT]:
                return False

            # 취소 상태로 변경
            success = self.task_registry.update_task_state(task_id, TaskState.CANCELLED)
            if success:
                self.logger.info(f"태스크 {task_id} 취소됨")

            return success

    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """태스크 상태 조회"""
        return self.task_registry.get_task_info(task_id)

    def list_active_tasks(self) -> List[TaskInfo]:
        """활성 태스크 목록"""
        return self.task_registry.list_active_tasks()

    def list_completed_tasks(self, max_age_seconds: float = 3600) -> List[TaskInfo]:
        """완료된 태스크 목록"""
        return self.task_registry.list_completed_tasks(max_age_seconds)

    def cleanup_completed_tasks(self, max_age: float = 3600.0) -> int:
        """완료된 태스크 정리"""
        cleaned_count = self.task_registry.cleanup_completed_tasks(max_age)
        if cleaned_count > 0:
            self.logger.info(f"정리된 완료 태스크: {cleaned_count}개")
        return cleaned_count

    def run_parallel(self, coroutines: List[Coroutine],
                     timeout: Optional[float] = None,
                     return_exceptions: bool = True) -> List[Any]:
        """여러 코루틴을 병렬로 실행"""
        if not coroutines:
            return []

        async def gather_coros():
            try:
                return await asyncio.gather(*coroutines, return_exceptions=return_exceptions)
            except Exception as e:
                self.logger.error(f"병렬 실행 오류: {e}")
                raise

        return self.run_async(gather_coros(), timeout)

    def run_with_semaphore(self, coroutines: List[Coroutine],
                           max_concurrent: int = 5,
                           timeout: Optional[float] = None) -> List[Any]:
        """세마포어를 사용한 제한된 병렬 실행"""

        async def limited_gather():
            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_with_sem(coro):
                async with semaphore:
                    return await coro

            tasks = [run_with_sem(coro) for coro in coroutines]
            return await asyncio.gather(*tasks, return_exceptions=True)

        return self.run_async(limited_gather(), timeout)

    @contextlib.asynccontextmanager
    async def managed_task(self, coro: Coroutine[Any, Any, T],
                           name: Optional[str] = None,
                           timeout: Optional[float] = None):
        """관리되는 태스크 컨텍스트"""
        task_id = None
        try:
            task_id = self.submit_coro(coro, name, timeout)
            result = self.get_task_result(task_id, timeout)
            yield result
        except Exception as e:
            if task_id:
                self.cancel_task(task_id)
            raise
        finally:
            # 태스크 정보는 자동 정리됨
            pass

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        active_tasks = self.list_active_tasks()
        completed_tasks = self.list_completed_tasks()

        # 실행자 상태
        executor_stats = {}

        with self._executor_lock:
            if self._thread_executor:
                executor_stats['thread_pool'] = {
                    'max_workers': self._thread_executor._max_workers,
                    'active_threads': len(self._thread_executor._threads)
                }
            if self._process_executor:
                executor_stats['process_pool'] = {
                    'max_workers': self._process_executor._max_workers
                }

        # 태스크 통계
        task_stats = {
            'active_count': len(active_tasks),
            'completed_count': len(completed_tasks),
            'active_by_type': defaultdict(int),
            'completed_by_state': defaultdict(int)
        }

        for task in active_tasks:
            task_stats['active_by_type'][task.executor_type.value] += 1

        for task in completed_tasks:
            task_stats['completed_by_state'][task.state.value] += 1

        return {
            'task_stats': dict(task_stats),
            'executor_stats': executor_stats,
            'loop_alive': not self.event_loop_manager._shutdown
        }

    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """매니저 종료"""
        with self._shutdown_lock:
            if self._shutdown:
                return

            self.logger.info("SafeAsyncManager 종료 시작...")
            self._shutdown = True

        try:
            # 1. 활성 태스크들 취소
            active_tasks = self.list_active_tasks()
            for task_info in active_tasks:
                self.cancel_task(task_info.task_id)

            if active_tasks:
                self.logger.info(f"{len(active_tasks)}개 활성 태스크 취소됨")

            # 2. 실행자들 종료
            with self._executor_lock:
                if self._thread_executor:
                    self._thread_executor.shutdown(wait=wait, timeout=timeout/2)
                    self._thread_executor = None
                    self.logger.debug("스레드 풀 종료됨")

                if self._process_executor:
                    self._process_executor.shutdown(wait=wait, timeout=timeout/2)
                    self._process_executor = None
                    self.logger.debug("프로세스 풀 종료됨")

            # 3. 태스크 레지스트리 종료
            self.task_registry.shutdown()
            self.logger.debug("태스크 레지스트리 종료됨")

            # 4. 이벤트 루프 종료
            self.event_loop_manager.stop_loop(timeout=timeout/2)
            self.logger.debug("이벤트 루프 종료됨")

            self.logger.info("SafeAsyncManager 종료 완료")

        except Exception as e:
            self.logger.error(f"SafeAsyncManager 종료 중 오류: {e}")

    def _emergency_shutdown(self):
        """긴급 종료 (atexit 콜백)"""
        try:
            self.shutdown(wait=False, timeout=5.0)
        except:
            pass

    def __del__(self):
        """소멸자"""
        self._emergency_shutdown()


# 전역 비동기 매니저 (스레드 안전)
_global_async_manager: Optional[SafeAsyncManager] = None
_async_manager_lock = threading.RLock()


def get_async_manager() -> SafeAsyncManager:
    """전역 비동기 매니저 반환 (싱글톤, 스레드 안전)"""
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


# 편의 함수들 (스레드 안전)
def run_async_safe(coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
    """안전한 비동기 실행"""
    manager = get_async_manager()
    return manager.run_async(coro, timeout)


def submit_async_task(coro: Coroutine, name: Optional[str] = None,
                     timeout: Optional[float] = None) -> str:
    """비동기 태스크 제출"""
    manager = get_async_manager()
    return manager.submit_coro(coro, name, timeout)


def run_parallel_safe(coroutines: List[Coroutine],
                     timeout: Optional[float] = None,
                     max_concurrent: Optional[int] = None) -> List[Any]:
    """안전한 병렬 실행"""
    manager = get_async_manager()

    if max_concurrent:
        return manager.run_with_semaphore(coroutines, max_concurrent, timeout)
    else:
        return manager.run_parallel(coroutines, timeout)


def get_async_stats() -> Dict[str, Any]:
    """비동기 성능 통계"""
    manager = get_async_manager()
    return manager.get_performance_stats()


# 데코레이터
def async_safe(timeout: Optional[float] = None, retry_count: int = 0):
    """비동기 안전 실행 데코레이터"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                coro = func(*args, **kwargs)

                # 재시도 로직
                last_exception = None
                for attempt in range(retry_count + 1):
                    try:
                        return run_async_safe(coro, timeout)
                    except Exception as e:
                        last_exception = e
                        if attempt < retry_count:
                            time.sleep(2 ** attempt)  # 지수 백오프
                            coro = func(*args, **kwargs)  # 새 코루틴 생성

                raise last_exception
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def concurrent_safe(max_concurrent: int = 5, timeout: Optional[float] = None):
    """동시성 제한 데코레이터"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(coroutines_or_args_list):
            if hasattr(coroutines_or_args_list[0], '__await__'):
                # 코루틴 리스트인 경우
                return run_parallel_safe(
                    coroutines_or_args_list,
                    timeout=timeout,
                    max_concurrent=max_concurrent
                )
            else:
                # 인자 리스트인 경우 - 함수를 여러 인자로 병렬 실행
                coroutines = [func(*args) if isinstance(args, (list, tuple)) else func(args)
                            for args in coroutines_or_args_list]
                return run_parallel_safe(
                    coroutines,
                    timeout=timeout,
                    max_concurrent=max_concurrent
                )

        return wrapper

    return decorator


# 프로그램 종료시 자동 정리
atexit.register(cleanup_async_manager)


# 사용 예시 및 테스트
if __name__ == "__main__":
    import time
    import random

    async def example_coro(delay: float, value: int, should_fail: bool = False):
        """예시 코루틴"""
        await asyncio.sleep(delay)
        if should_fail:
            raise ValueError(f"의도적 실패: {value}")
        return value * 2

    def example_func(delay: float, value: int, should_fail: bool = False):
        """예시 함수"""
        time.sleep(delay)
        if should_fail:
            raise ValueError(f"의도적 실패: {value}")
        return value * 3

    async def test_concurrent_access():
        """동시 접근 테스트"""
        manager = get_async_manager()

        # 여러 스레드에서 동시에 태스크 제출
        import threading

        results = []
        exceptions = []

        def submit_tasks(thread_id: int):
            try:
                for i in range(5):
                    task_id = manager.submit_coro(
                        example_coro(0.1, thread_id * 10 + i),
                        name=f"thread_{thread_id}_task_{i}"
                    )
                    result = manager.get_task_result(task_id, timeout=5.0)
                    results.append(result)
            except Exception as e:
                exceptions.append(e)

        # 5개 스레드에서 동시 실행
        threads = []
        for i in range(5):
            thread = threading.Thread(target=submit_tasks, args=(i,))
            threads.append(thread)
            thread.start()

        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()

        return results, exceptions

    def main():
        print("=== 개선된 비동기 관리 시스템 테스트 ===")

        # 비동기 매니저 테스트
        manager = get_async_manager()

        # 1. 단일 코루틴 실행
        print("\n1. 단일 코루틴 실행 테스트")
        result = manager.run_async(example_coro(0.1, 5))
        print(f"   결과: {result}")

        # 2. 태스크 제출 및 결과 대기
        print("\n2. 태스크 제출 테스트")
        task_id = manager.submit_func(
            example_func, 0.1, 10,
            executor_type=ExecutorType.THREAD,
            name="test_function"
        )
        task_result = manager.get_task_result(task_id, timeout=5.0)
        print(f"   태스크 결과: {task_result}")

        # 3. 병렬 실행
        print("\n3. 병렬 실행 테스트")
        coroutines = [example_coro(0.1, i) for i in range(3)]
        parallel_results = manager.run_parallel(coroutines)
        print(f"   병렬 결과: {parallel_results}")

        # 4. 제한된 병렬 실행
        print("\n4. 제한된 병렬 실행 테스트")
        many_coros = [example_coro(0.05, i) for i in range(10)]
        limited_results = manager.run_with_semaphore(many_coros, max_concurrent=3)
        print(f"   제한된 병렬 결과: {len(limited_results)}개")

        # 5. 동시성 안전성 테스트
        print("\n5. 동시성 안전성 테스트")
        concurrent_results, concurrent_exceptions = run_async_safe(test_concurrent_access())
        print(f"   동시 실행 결과: {len(concurrent_results)}개 성공, {len(concurrent_exceptions)}개 실패")

        # 6. 성능 통계
        print("\n6. 성능 통계")
        stats = manager.get_performance_stats()
        print(f"   활성 태스크: {stats['task_stats']['active_count']}개")
        print(f"   완료 태스크: {stats['task_stats']['completed_count']}개")
        print(f"   이벤트 루프 상태: {'활성' if stats['loop_alive'] else '비활성'}")

        # 7. 오류 처리 테스트
        print("\n7. 오류 처리 테스트")
        try:
            error_task_id = manager.submit_coro(
                example_coro(0.1, 5, should_fail=True),
                name="error_test"
            )
            manager.get_task_result(error_task_id, timeout=5.0)
        except ValueError as e:
            print(f"   예상된 오류 정상 처리: {e}")

        # 8. 타임아웃 테스트
        print("\n8. 타임아웃 테스트")
        try:
            timeout_task_id = manager.submit_coro(
                example_coro(2.0, 5),  # 2초 실행
                name="timeout_test",
                timeout=0.5  # 0.5초 타임아웃
            )
            manager.get_task_result(timeout_task_id, timeout=1.0)
        except TimeoutError:
            print("   타임아웃 정상 처리됨")

        # 9. 태스크 정리
        print("\n9. 태스크 정리 테스트")
        cleaned_count = manager.cleanup_completed_tasks(max_age=1.0)
        print(f"   정리된 태스크: {cleaned_count}개")

        # 10. 데코레이터 테스트
        print("\n10. 데코레이터 테스트")

        @async_safe(timeout=2.0, retry_count=2)
        async def decorated_coro(value: int):
            if random.random() < 0.3:  # 30% 확률로 실패
                raise RuntimeError("랜덤 실패")
            await asyncio.sleep(0.1)
            return value * 2

        try:
            decorated_result = decorated_coro(42)
            print(f"   데코레이터 결과: {decorated_result}")
        except Exception as e:
            print(f"   데코레이터 오류: {e}")

        print("\n✅ 개선된 비동기 관리 시스템 테스트 완료")

        # 최종 정리
        cleanup_async_manager()

    if __name__ == "__main__":
        main()