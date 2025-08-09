"""
개선된 오류 처리 시스템
Important 문제 해결: 상세한 예외 처리 및 로깅 추가
"""
import logging
import traceback
import functools
import threading
import time
from typing import Dict, List, Optional, Callable, Any, Type
from dataclasses import dataclass, field
from enum import Enum
import contextlib
import sys


class ErrorSeverity(Enum):
    """오류 심각도"""
    LOW = "low"  # 경고, 계속 진행 가능
    MEDIUM = "medium"  # 오류, 복구 시도 가능
    HIGH = "high"  # 심각한 오류, 즉시 처리 필요
    CRITICAL = "critical"  # 치명적 오류, 시스템 중단


class ErrorCategory(Enum):
    """오류 카테고리"""
    MEMORY = "memory"
    MODEL_LOADING = "model_loading"
    INFERENCE = "inference"
    CONFIGURATION = "configuration"
    HARDWARE = "hardware"
    NETWORK = "network"
    DATA = "data"
    PERMISSION = "permission"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """오류 정보"""
    id: str
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception_type: str
    exception_message: str
    traceback_str: str
    context: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    retry_count: int = 0
    resolved: bool = False


class ErrorPattern:
    """오류 패턴 정의"""

    def __init__(self, pattern: str, category: ErrorCategory,
                 severity: ErrorSeverity, suggestions: List[str]):
        self.pattern = pattern.lower()
        self.category = category
        self.severity = severity
        self.suggestions = suggestions

    def matches(self, error_message: str) -> bool:
        """오류 메시지가 패턴과 일치하는지 확인"""
        return self.pattern in error_message.lower()


class ErrorPatternRegistry:
    """오류 패턴 레지스트리"""

    def __init__(self):
        self.patterns: List[ErrorPattern] = []
        self._setup_default_patterns()

    def _setup_default_patterns(self):
        """기본 오류 패턴들 설정"""
        # 메모리 관련 오류
        self.patterns.extend([
            ErrorPattern(
                "cuda out of memory",
                ErrorCategory.MEMORY,
                ErrorSeverity.HIGH,
                [
                    "배치 크기를 줄여보세요",
                    "4-bit 양자화를 활성화하세요 (load_in_4bit=True)",
                    "더 작은 모델을 사용해보세요",
                    "GPU 메모리를 정리하세요 (torch.cuda.empty_cache())"
                ]
            ),
            ErrorPattern(
                "out of memory",
                ErrorCategory.MEMORY,
                ErrorSeverity.HIGH,
                [
                    "메모리 사용량을 확인하세요",
                    "불필요한 프로세스를 종료하세요",
                    "가상 메모리를 늘려보세요"
                ]
            ),
            ErrorPattern(
                "memory_allocated",
                ErrorCategory.MEMORY,
                ErrorSeverity.MEDIUM,
                ["메모리 모니터링을 확인하세요"]
            )
        ])

        # 모델 로딩 관련 오류
        self.patterns.extend([
            ErrorPattern(
                "trust_remote_code",
                ErrorCategory.MODEL_LOADING,
                ErrorSeverity.MEDIUM,
                [
                    "신뢰할 수 있는 모델인지 확인하세요",
                    "trust_remote_code=True를 설정하세요 (보안 위험 주의)"
                ]
            ),
            ErrorPattern(
                "model not found",
                ErrorCategory.MODEL_LOADING,
                ErrorSeverity.MEDIUM,
                [
                    "모델 경로를 확인하세요",
                    "Hugging Face Hub에서 모델이 존재하는지 확인하세요",
                    "인터넷 연결을 확인하세요"
                ]
            ),
            ErrorPattern(
                "repository not found",
                ErrorCategory.MODEL_LOADING,
                ErrorSeverity.MEDIUM,
                [
                    "모델 저장소 이름을 확인하세요",
                    "모델이 공개되어 있는지 확인하세요",
                    "Hugging Face 토큰이 필요한지 확인하세요"
                ]
            ),
            ErrorPattern(
                "403 forbidden",
                ErrorCategory.PERMISSION,
                ErrorSeverity.MEDIUM,
                [
                    "모델 접근 권한을 확인하세요",
                    "Hugging Face 토큰을 설정하세요",
                    "모델 라이선스를 확인하세요"
                ]
            )
        ])

        # 추론 관련 오류
        self.patterns.extend([
            ErrorPattern(
                "already borrowed",
                ErrorCategory.INFERENCE,
                ErrorSeverity.MEDIUM,
                [
                    "스레드 안전성 문제입니다",
                    "TOKENIZERS_PARALLELISM=false 환경변수를 설정하세요",
                    "순차 처리로 변경하세요"
                ]
            ),
            ErrorPattern(
                "signal only works in main thread",
                ErrorCategory.INFERENCE,
                ErrorSeverity.LOW,
                [
                    "메인 스레드가 아닌 곳에서 신호 처리를 시도했습니다",
                    "멀티스레딩 환경에서는 신호 처리를 비활성화하세요"
                ]
            ),
            ErrorPattern(
                "device-side assert triggered",
                ErrorCategory.HARDWARE,
                ErrorSeverity.HIGH,
                [
                    "CUDA 장치 오류입니다",
                    "GPU 드라이버를 업데이트하세요",
                    "CUDA 버전 호환성을 확인하세요"
                ]
            )
        ])

        # 의존성 관련 오류
        self.patterns.extend([
            ErrorPattern(
                "no module named",
                ErrorCategory.DEPENDENCY,
                ErrorSeverity.MEDIUM,
                [
                    "필요한 패키지를 설치하세요",
                    "pip install 명령어를 사용하세요",
                    "가상환경이 활성화되어 있는지 확인하세요"
                ]
            ),
            ErrorPattern(
                "import error",
                ErrorCategory.DEPENDENCY,
                ErrorSeverity.MEDIUM,
                [
                    "패키지 설치를 확인하세요",
                    "패키지 버전 호환성을 확인하세요"
                ]
            )
        ])

        # 설정 관련 오류
        self.patterns.extend([
            ErrorPattern(
                "configuration",
                ErrorCategory.CONFIGURATION,
                ErrorSeverity.MEDIUM,
                [
                    "설정 파일 형식을 확인하세요",
                    "필수 설정 항목이 누락되지 않았는지 확인하세요"
                ]
            ),
            ErrorPattern(
                "invalid choice",
                ErrorCategory.CONFIGURATION,
                ErrorSeverity.MEDIUM,
                [
                    "설정값이 허용된 범위 내에 있는지 확인하세요",
                    "문서에서 유효한 옵션들을 확인하세요"
                ]
            )
        ])

    def find_pattern(self, error_message: str) -> Optional[ErrorPattern]:
        """오류 메시지에 맞는 패턴 찾기"""
        for pattern in self.patterns:
            if pattern.matches(error_message):
                return pattern
        return None

    def add_pattern(self, pattern: ErrorPattern):
        """새로운 패턴 추가"""
        self.patterns.append(pattern)


class ErrorHandler:
    """통합 오류 처리기"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.error_history: List[ErrorInfo] = []
        self.pattern_registry = ErrorPatternRegistry()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._error_counter = 0

        # 자동 복구 설정
        self.auto_recovery_enabled = True
        self.max_retry_attempts = 3

        # 콜백 함수들
        self.error_callbacks: Dict[ErrorSeverity, List[Callable]] = {
            severity: [] for severity in ErrorSeverity
        }

    def handle_exception(self, exception: Exception,
                         context: Optional[Dict[str, Any]] = None,
                         severity: Optional[ErrorSeverity] = None) -> ErrorInfo:
        """예외 처리"""
        with self._lock:
            self._error_counter += 1
            error_id = f"error_{self._error_counter}_{int(time.time())}"

        # 오류 정보 수집
        error_message = str(exception)
        exception_type = type(exception).__name__
        traceback_str = traceback.format_exc()

        # 패턴 매칭으로 카테고리 및 심각도 결정
        pattern = self.pattern_registry.find_pattern(error_message)

        if pattern:
            category = pattern.category
            severity = severity or pattern.severity
            suggestions = pattern.suggestions.copy()
        else:
            category = self._infer_category(exception, error_message)
            severity = severity or self._infer_severity(exception, error_message)
            suggestions = self._generate_generic_suggestions(exception)

        # 오류 정보 생성
        error_info = ErrorInfo(
            id=error_id,
            timestamp=time.time(),
            category=category,
            severity=severity,
            message=error_message,
            exception_type=exception_type,
            exception_message=error_message,
            traceback_str=traceback_str,
            context=context or {},
            suggestions=suggestions
        )

        # 히스토리에 추가
        self._add_to_history(error_info)

        # 로깅
        self._log_error(error_info)

        # 콜백 실행
        self._execute_callbacks(error_info)

        # 자동 복구 시도
        if self.auto_recovery_enabled:
            self._attempt_auto_recovery(error_info)

        return error_info

    def _infer_category(self, exception: Exception, message: str) -> ErrorCategory:
        """예외 타입과 메시지로 카테고리 추론"""
        exception_type = type(exception).__name__
        message_lower = message.lower()

        if "memory" in message_lower or "cuda" in message_lower:
            return ErrorCategory.MEMORY
        elif isinstance(exception, (ImportError, ModuleNotFoundError)):
            return ErrorCategory.DEPENDENCY
        elif isinstance(exception, (PermissionError, OSError)):
            return ErrorCategory.PERMISSION
        elif "config" in message_lower or "setting" in message_lower:
            return ErrorCategory.CONFIGURATION
        elif "model" in message_lower or "load" in message_lower:
            return ErrorCategory.MODEL_LOADING
        elif "network" in message_lower or "connection" in message_lower:
            return ErrorCategory.NETWORK
        else:
            return ErrorCategory.UNKNOWN

    def _infer_severity(self, exception: Exception, message: str) -> ErrorSeverity:
        """예외 타입과 메시지로 심각도 추론"""
        message_lower = message.lower()

        if ("cuda out of memory" in message_lower or
                "device-side assert" in message_lower):
            return ErrorSeverity.CRITICAL
        elif ("memory" in message_lower or
              "permission denied" in message_lower):
            return ErrorSeverity.HIGH
        elif isinstance(exception, (ImportError, ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _generate_generic_suggestions(self, exception: Exception) -> List[str]:
        """일반적인 해결책 제안"""
        exception_type = type(exception).__name__

        suggestions = []

        if isinstance(exception, MemoryError):
            suggestions.extend([
                "메모리 사용량을 줄여보세요",
                "불필요한 프로세스를 종료하세요",
                "배치 크기를 줄여보세요"
            ])
        elif isinstance(exception, ImportError):
            suggestions.extend([
                "필요한 패키지를 설치하세요",
                "패키지 버전을 확인하세요"
            ])
        elif isinstance(exception, FileNotFoundError):
            suggestions.extend([
                "파일 경로를 확인하세요",
                "파일이 존재하는지 확인하세요"
            ])
        elif isinstance(exception, PermissionError):
            suggestions.extend([
                "파일/폴더 권한을 확인하세요",
                "관리자 권한으로 실행해보세요"
            ])
        else:
            suggestions.append("문서를 참조하거나 로그를 확인하세요")

        return suggestions

    def _add_to_history(self, error_info: ErrorInfo):
        """히스토리에 오류 추가"""
        with self._lock:
            self.error_history.append(error_info)

            # 히스토리 크기 제한
            if len(self.error_history) > self.max_history:
                self.error_history = self.error_history[-self.max_history:]

    def _log_error(self, error_info: ErrorInfo):
        """오류 로깅"""
        log_message = (
            f"[{error_info.severity.value.upper()}] "
            f"{error_info.category.value}: {error_info.message}"
        )

        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

        # 제안사항 로깅
        if error_info.suggestions:
            self.logger.info(f"해결 제안: {', '.join(error_info.suggestions)}")

    def _execute_callbacks(self, error_info: ErrorInfo):
        """콜백 함수 실행"""
        callbacks = self.error_callbacks.get(error_info.severity, [])

        for callback in callbacks:
            try:
                callback(error_info)
            except Exception as e:
                self.logger.error(f"오류 콜백 실행 실패: {e}")

    def _attempt_auto_recovery(self, error_info: ErrorInfo):
        """자동 복구 시도"""
        if error_info.category == ErrorCategory.MEMORY:
            self._recover_memory_error(error_info)
        elif error_info.category == ErrorCategory.MODEL_LOADING:
            self._recover_model_loading_error(error_info)

    def _recover_memory_error(self, error_info: ErrorInfo):
        """메모리 오류 복구"""
        try:
            import gc
            import torch

            # 가비지 컬렉션
            gc.collect()

            # CUDA 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            error_info.context['auto_recovery'] = 'memory_cleanup_attempted'
            self.logger.info("메모리 정리 자동 복구 시도 완료")

        except Exception as e:
            self.logger.error(f"메모리 복구 실패: {e}")

    def _recover_model_loading_error(self, error_info: ErrorInfo):
        """모델 로딩 오류 복구"""
        # 실제 구현에서는 더 구체적인 복구 로직 필요
        error_info.context['auto_recovery'] = 'model_loading_fallback_suggested'
        self.logger.info("모델 로딩 오류 복구 제안 생성")

    def add_error_callback(self, severity: ErrorSeverity, callback: Callable[[ErrorInfo], None]):
        """오류 콜백 추가"""
        self.error_callbacks[severity].append(callback)

    def get_error_history(self, category: Optional[ErrorCategory] = None,
                          severity: Optional[ErrorSeverity] = None,
                          limit: Optional[int] = None) -> List[ErrorInfo]:
        """오류 히스토리 조회"""
        with self._lock:
            filtered_errors = self.error_history.copy()

        # 필터링
        if category:
            filtered_errors = [e for e in filtered_errors if e.category == category]

        if severity:
            filtered_errors = [e for e in filtered_errors if e.severity == severity]

        # 최신순 정렬
        filtered_errors.sort(key=lambda x: x.timestamp, reverse=True)

        # 제한
        if limit:
            filtered_errors = filtered_errors[:limit]

        return filtered_errors

    def get_error_stats(self) -> Dict[str, int]:
        """오류 통계"""
        with self._lock:
            stats = {
                'total_errors': len(self.error_history),
                'by_category': {},
                'by_severity': {},
                'resolved_count': sum(1 for e in self.error_history if e.resolved)
            }

            for error in self.error_history:
                # 카테고리별
                category = error.category.value
                stats['by_category'][category] = stats['by_category'].get(category, 0) + 1

                # 심각도별
                severity = error.severity.value
                stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

        return stats

    def clear_history(self):
        """히스토리 정리"""
        with self._lock:
            self.error_history.clear()
            self._error_counter = 0


# 데코레이터 함수들
def safe_execute(error_handler: Optional[ErrorHandler] = None,
                 fallback_result: Any = None,
                 re_raise: bool = False):
    """안전 실행 데코레이터"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or get_global_error_handler()

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = handler.handle_exception(
                    e,
                    context={'function': func.__name__, 'args': str(args)[:100]}
                )

                if re_raise:
                    raise
                else:
                    return fallback_result

        return wrapper

    return decorator


def async_safe_execute(error_handler: Optional[ErrorHandler] = None,
                       fallback_result: Any = None,
                       re_raise: bool = False):
    """비동기 안전 실행 데코레이터"""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            handler = error_handler or get_global_error_handler()

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_info = handler.handle_exception(
                    e,
                    context={'async_function': func.__name__, 'args': str(args)[:100]}
                )

                if re_raise:
                    raise
                else:
                    return fallback_result

        return wrapper

    return decorator


@contextlib.contextmanager
def error_context(handler: Optional[ErrorHandler] = None,
                  context_info: Optional[Dict[str, Any]] = None,
                  suppress: bool = False):
    """오류 컨텍스트 관리자"""
    error_handler = handler or get_global_error_handler()

    try:
        yield
    except Exception as e:
        error_info = error_handler.handle_exception(e, context=context_info)

        if not suppress:
            raise


# 전역 오류 처리기
_global_error_handler: Optional[ErrorHandler] = None
_handler_lock = threading.Lock()


def get_global_error_handler() -> ErrorHandler:
    """전역 오류 처리기 반환"""
    global _global_error_handler

    with _handler_lock:
        if _global_error_handler is None:
            _global_error_handler = ErrorHandler()

            # 기본 콜백 설정
            def critical_error_callback(error_info: ErrorInfo):
                print(f"🚨 CRITICAL ERROR: {error_info.message}")
                if error_info.suggestions:
                    print(f"💡 제안: {', '.join(error_info.suggestions)}")

            _global_error_handler.add_error_callback(
                ErrorSeverity.CRITICAL,
                critical_error_callback
            )

    return _global_error_handler


def cleanup_error_handler():
    """전역 오류 처리기 정리"""
    global _global_error_handler

    with _handler_lock:
        if _global_error_handler is not None:
            _global_error_handler.clear_history()
            _global_error_handler = None


# 편의 함수들
def handle_error(exception: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorInfo:
    """편의 오류 처리 함수"""
    return get_global_error_handler().handle_exception(exception, context)


def get_recent_errors(limit: int = 10) -> List[ErrorInfo]:
    """최근 오류 조회"""
    return get_global_error_handler().get_error_history(limit=limit)


def get_error_statistics() -> Dict[str, int]:
    """오류 통계 조회"""
    return get_global_error_handler().get_error_stats()


# 프로그램 종료시 자동 정리
import atexit

atexit.register(cleanup_error_handler)

# 사용 예시
if __name__ == "__main__":
    print("=== 오류 처리 시스템 테스트 ===")

    # 오류 처리기 초기화
    handler = get_global_error_handler()

    # 1. 기본 오류 처리
    try:
        raise ValueError("테스트 오류 메시지")
    except Exception as e:
        error_info = handler.handle_exception(e, context={'test': 'basic_error'})
        print(f"오류 ID: {error_info.id}")
        print(f"카테고리: {error_info.category.value}")
        print(f"심각도: {error_info.severity.value}")
        print(f"제안사항: {error_info.suggestions}")

    # 2. 메모리 오류 시뮬레이션
    try:
        raise RuntimeError("CUDA out of memory")
    except Exception as e:
        error_info = handler.handle_exception(e)
        print(f"\n메모리 오류 감지:")
        print(f"  카테고리: {error_info.category.value}")
        print(f"  제안사항: {error_info.suggestions}")


    # 3. 데코레이터 사용
    @safe_execute(fallback_result="fallback")
    def risky_function():
        raise ImportError("No module named 'test'")


    result = risky_function()
    print(f"\n데코레이터 결과: {result}")

    # 4. 컨텍스트 매니저 사용
    with error_context(context_info={'operation': 'test_context'}, suppress=True):
        raise FileNotFoundError("test file not found")

    print("컨텍스트 매니저에서 오류가 억제됨")

    # 5. 통계 조회
    stats = handler.get_error_stats()
    print(f"\n오류 통계: {stats}")

    # 6. 최근 오류 조회
    recent = handler.get_error_history(limit=3)
    print(f"\n최근 오류 {len(recent)}개:")
    for error in recent:
        print(f"  {error.category.value}: {error.message[:50]}...")

    # 정리
    cleanup_error_handler()
    print("\n✅ 오류 처리 시스템 테스트 완료")