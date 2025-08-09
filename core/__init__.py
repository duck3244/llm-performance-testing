"""
핵심 시스템 모듈 초기화
"""
from .memory_manager import (
    ImprovedResourceManager,
    MemoryLevel,
    MemoryStats,
    get_resource_manager,
    cleanup_all_resources
)

from .async_manager import (
    SafeAsyncManager,
    ExecutorType,
    TaskInfo,
    get_async_manager,
    cleanup_async_manager,
    run_async_safe,
    submit_async_task,
    run_parallel_safe,
    async_safe
)

from .error_handler import (
    ErrorSeverity,
    ErrorCategory,
    ErrorInfo,
    ErrorPattern,
    ErrorHandler,
    get_global_error_handler,
    cleanup_error_handler,
    safe_execute,
    async_safe_execute,
    error_context,
    handle_error,
    get_recent_errors,
    get_error_statistics
)

from .improved_optimizer import (
    InferenceParams,
    OptimizationResult,
    SafeOptimizer,
    quick_optimize,
    safe_grid_search
)

__all__ = [
    # Memory manager
    'ImprovedResourceManager',
    'MemoryLevel',
    'MemoryStats',
    'get_resource_manager',
    'cleanup_all_resources',

    # Async manager
    'SafeAsyncManager',
    'ExecutorType',
    'TaskInfo',
    'get_async_manager',
    'cleanup_async_manager',
    'run_async_safe',
    'submit_async_task',
    'run_parallel_safe',
    'async_safe',

    # Error handler
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorInfo',
    'ErrorPattern',
    'ErrorHandler',
    'get_global_error_handler',
    'cleanup_error_handler',
    'safe_execute',
    'async_safe_execute',
    'error_context',
    'handle_error',
    'get_recent_errors',
    'get_error_statistics',

    # Optimizer
    'InferenceParams',
    'OptimizationResult',
    'SafeOptimizer',
    'quick_optimize',
    'safe_grid_search'
]

# 버전 정보
__version__ = "2.0.0"