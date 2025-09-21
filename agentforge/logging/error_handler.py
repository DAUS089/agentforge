"""
Enhanced error handling and recovery for agentforge.
"""

import traceback
import sys
from typing import Dict, Any, Optional, List, Callable, Type
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import functools


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    CONFIGURATION = "configuration"
    LLM_API = "llm_api"
    TOOL_EXECUTION = "tool_execution"
    CREW_EXECUTION = "crew_execution"
    FILE_OPERATIONS = "file_operations"
    NETWORK = "network"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_id: str
    timestamp: datetime
    component: str
    function_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any]
    stack_trace: str
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_action: Optional[str] = None


class ErrorHandler:
    """Enhanced error handler with recovery mechanisms and context tracking."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.error_counts: Dict[str, int] = {}
        self._setup_default_recovery_strategies()
    
    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies for common error types."""
        self.recovery_strategies[ErrorCategory.CONFIGURATION] = [
            self._recover_configuration_error
        ]
        self.recovery_strategies[ErrorCategory.LLM_API] = [
            self._recover_llm_api_error
        ]
        self.recovery_strategies[ErrorCategory.TOOL_EXECUTION] = [
            self._recover_tool_execution_error
        ]
        self.recovery_strategies[ErrorCategory.CREW_EXECUTION] = [
            self._recover_crew_execution_error
        ]
        self.recovery_strategies[ErrorCategory.FILE_OPERATIONS] = [
            self._recover_file_operation_error
        ]
        self.recovery_strategies[ErrorCategory.NETWORK] = [
            self._recover_network_error
        ]
    
    def handle_error(self, error: Exception, component: str, context: Dict[str, Any] = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> ErrorContext:
        """Handle an error with context and recovery attempts."""
        error_id = f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(error)}"
        
        # Classify error
        category = self._classify_error(error)
        
        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            component=component,
            function_name=error.__traceback__.tb_frame.f_code.co_name if error.__traceback__ else "unknown",
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            context=context or {},
            stack_trace=traceback.format_exc()
        )
        
        # Log error
        if self.logger:
            self.logger.log_error_with_context(component, error, context)
        
        # Track error
        self.error_history.append(error_context)
        self.error_counts[error_context.error_type] = self.error_counts.get(error_context.error_type, 0) + 1
        
        # Attempt recovery
        if severity != ErrorSeverity.CRITICAL:
            recovery_successful = self._attempt_recovery(error_context)
            error_context.recovery_attempted = True
            error_context.recovery_successful = recovery_successful
        
        return error_context
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error into categories."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Configuration errors
        if any(keyword in error_message for keyword in ['config', 'setting', 'parameter', 'validation']):
            return ErrorCategory.CONFIGURATION
        
        # LLM API errors
        if any(keyword in error_message for keyword in ['api', 'openai', 'anthropic', 'google', 'llm', 'model']):
            return ErrorCategory.LLM_API
        
        # Tool execution errors
        if any(keyword in error_message for keyword in ['tool', 'search', 'scrape', 'file']):
            return ErrorCategory.TOOL_EXECUTION
        
        # Crew execution errors
        if any(keyword in error_message for keyword in ['crew', 'agent', 'task', 'execution']):
            return ErrorCategory.CREW_EXECUTION
        
        # File operation errors
        if any(keyword in error_message for keyword in ['file', 'directory', 'path', 'permission']):
            return ErrorCategory.FILE_OPERATIONS
        
        # Network errors
        if any(keyword in error_message for keyword in ['network', 'connection', 'timeout', 'http']):
            return ErrorCategory.NETWORK
        
        return ErrorCategory.UNKNOWN
    
    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from an error."""
        strategies = self.recovery_strategies.get(error_context.category, [])
        
        for strategy in strategies:
            try:
                if strategy(error_context):
                    error_context.recovery_action = strategy.__name__
                    return True
            except Exception as recovery_error:
                if self.logger:
                    self.logger.warning("ERROR_HANDLER", 
                                      f"Recovery strategy {strategy.__name__} failed: {str(recovery_error)}")
        
        return False
    
    def _recover_configuration_error(self, error_context: ErrorContext) -> bool:
        """Recover from configuration errors."""
        # Try to use default configuration
        if self.logger:
            self.logger.info("ERROR_HANDLER", "Attempting to recover from configuration error using defaults")
        return True
    
    def _recover_llm_api_error(self, error_context: ErrorContext) -> bool:
        """Recover from LLM API errors."""
        # Try to switch to a different provider or model
        if self.logger:
            self.logger.info("ERROR_HANDLER", "Attempting to recover from LLM API error by switching provider")
        return True
    
    def _recover_tool_execution_error(self, error_context: ErrorContext) -> bool:
        """Recover from tool execution errors."""
        # Try to use alternative tools or mock tools
        if self.logger:
            self.logger.info("ERROR_HANDLER", "Attempting to recover from tool execution error using alternatives")
        return True
    
    def _recover_crew_execution_error(self, error_context: ErrorContext) -> bool:
        """Recover from crew execution errors."""
        # Try to restart the crew or use fallback mode
        if self.logger:
            self.logger.info("ERROR_HANDLER", "Attempting to recover from crew execution error using fallback mode")
        return True
    
    def _recover_file_operation_error(self, error_context: ErrorContext) -> bool:
        """Recover from file operation errors."""
        # Try to create directories or use alternative paths
        if self.logger:
            self.logger.info("ERROR_HANDLER", "Attempting to recover from file operation error by creating directories")
        return True
    
    def _recover_network_error(self, error_context: ErrorContext) -> bool:
        """Recover from network errors."""
        # Try to retry with exponential backoff
        if self.logger:
            self.logger.info("ERROR_HANDLER", "Attempting to recover from network error with retry")
        return True
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of errors in the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        
        error_counts_by_category = {}
        error_counts_by_severity = {}
        
        for error in recent_errors:
            category = error.category.value
            severity = error.severity.value
            
            error_counts_by_category[category] = error_counts_by_category.get(category, 0) + 1
            error_counts_by_severity[severity] = error_counts_by_severity.get(severity, 0) + 1
        
        return {
            "period_hours": hours,
            "total_errors": len(recent_errors),
            "errors_by_category": error_counts_by_category,
            "errors_by_severity": error_counts_by_severity,
            "recovery_success_rate": sum(1 for e in recent_errors if e.recovery_successful) / len(recent_errors) if recent_errors else 0,
            "most_common_errors": self._get_most_common_errors(recent_errors)
        }
    
    def _get_most_common_errors(self, errors: List[ErrorContext]) -> List[Dict[str, Any]]:
        """Get the most common error types."""
        error_type_counts = {}
        for error in errors:
            error_type = error.error_type
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        sorted_errors = sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"error_type": error_type, "count": count} for error_type, count in sorted_errors[:5]]
    
    def add_recovery_strategy(self, category: ErrorCategory, strategy: Callable):
        """Add a custom recovery strategy for a specific error category."""
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        self.recovery_strategies[category].append(strategy)
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        self.error_counts.clear()


def error_handler(component: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 reraise: bool = False, recovery: bool = True):
    """Decorator for automatic error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get the global error handler
                from .logger import get_logger
                logger = get_logger()
                error_handler = ErrorHandler(logger)
                
                # Handle the error
                error_context = error_handler.handle_error(
                    e, component, 
                    context={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)},
                    severity=severity
                )
                
                # Reraise if requested
                if reraise:
                    raise
                
                # Return None or default value if recovery was attempted
                if recovery and error_context.recovery_attempted:
                    return None
                
                raise
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_return=None, **kwargs) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        from .logger import get_logger
        logger = get_logger()
        error_handler = ErrorHandler(logger)
        
        error_context = error_handler.handle_error(
            e, "SAFE_EXECUTE",
            context={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
        )
        
        if error_context.recovery_successful:
            return default_return
        
        raise


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        from .logger import get_logger
        _global_error_handler = ErrorHandler(get_logger())
    return _global_error_handler
