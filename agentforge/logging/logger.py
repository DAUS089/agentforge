"""
Enhanced logging system for agentforge.
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict
import traceback


class LogLevel(Enum):
    """Log levels for agentforge."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    component: str
    message: str
    context: Dict[str, Any]
    trace_id: Optional[str] = None
    execution_id: Optional[str] = None
    crew_name: Optional[str] = None
    agent_name: Optional[str] = None
    task_name: Optional[str] = None


class AgentForgeLogger:
    """Enhanced logger for AgentForge with structured logging and context tracking."""
    
    def __init__(self, name: str = "agentforge", log_level: LogLevel = LogLevel.INFO,
                 log_file: Optional[str] = None, enable_console: bool = True):
        self.name = name
        self.log_level = log_level
        self.log_file = log_file
        self.enable_console = enable_console
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.value))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters
        self.console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(component)-15s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        self.file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(component)-15s | %(trace_id)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.console_formatter)
            self.logger.addHandler(console_handler)
        
        # Setup file handler
        if log_file:
            self._setup_file_handler(log_file)
        
        # Context tracking
        self._context_stack: List[Dict[str, Any]] = []
        self._current_trace_id: Optional[str] = None
        self._current_execution_id: Optional[str] = None
    
    def _setup_file_handler(self, log_file: str):
        """Setup file handler for logging."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)
    
    def set_log_level(self, level: LogLevel):
        """Set the log level."""
        self.log_level = level
        self.logger.setLevel(getattr(logging, level.value))
    
    def set_trace_id(self, trace_id: str):
        """Set the current trace ID for request tracking."""
        self._current_trace_id = trace_id
    
    def set_execution_id(self, execution_id: str):
        """Set the current execution ID for crew tracking."""
        self._current_execution_id = execution_id
    
    def push_context(self, **context):
        """Push context information onto the stack."""
        self._context_stack.append(context)
    
    def pop_context(self):
        """Pop the most recent context from the stack."""
        if self._context_stack:
            return self._context_stack.pop()
        return {}
    
    def clear_context(self):
        """Clear all context information."""
        self._context_stack.clear()
    
    def _create_log_entry(self, level: str, component: str, message: str, 
                         context: Dict[str, Any] = None) -> LogEntry:
        """Create a structured log entry."""
        # Merge all context information
        merged_context = {}
        for ctx in self._context_stack:
            merged_context.update(ctx)
        if context:
            merged_context.update(context)
        
        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            component=component,
            message=message,
            context=merged_context,
            trace_id=self._current_trace_id,
            execution_id=self._current_execution_id,
            crew_name=merged_context.get('crew_name'),
            agent_name=merged_context.get('agent_name'),
            task_name=merged_context.get('task_name')
        )
    
    def _log(self, level: str, component: str, message: str, context: Dict[str, Any] = None,
             exc_info: bool = False):
        """Internal logging method."""
        log_entry = self._create_log_entry(level, component, message, context)
        
        # Create log record
        record = logging.LogRecord(
            name=self.name,
            level=getattr(logging, level),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=exc_info
        )
        
        # Add custom attributes
        record.component = component
        record.trace_id = log_entry.trace_id or "N/A"
        record.execution_id = log_entry.execution_id or "N/A"
        record.crew_name = log_entry.crew_name or "N/A"
        record.agent_name = log_entry.agent_name or "N/A"
        record.task_name = log_entry.task_name or "N/A"
        
        # Add context as JSON
        record.context = json.dumps(log_entry.context, default=str)
        
        # Log the record
        self.logger.handle(record)
    
    def debug(self, component: str, message: str, context: Dict[str, Any] = None):
        """Log debug message."""
        self._log("DEBUG", component, message, context)
    
    def info(self, component: str, message: str, context: Dict[str, Any] = None):
        """Log info message."""
        self._log("INFO", component, message, context)
    
    def warning(self, component: str, message: str, context: Dict[str, Any] = None):
        """Log warning message."""
        self._log("WARNING", component, message, context)
    
    def error(self, component: str, message: str, context: Dict[str, Any] = None, 
              exc_info: bool = False):
        """Log error message."""
        self._log("ERROR", component, message, context, exc_info)
    
    def critical(self, component: str, message: str, context: Dict[str, Any] = None,
                 exc_info: bool = False):
        """Log critical message."""
        self._log("CRITICAL", component, message, context, exc_info)
    
    def log_ai_decision(self, agent: str, decision: str, reasoning: str, 
                       context: Dict[str, Any] = None):
        """Log AI decision-making process."""
        decision_context = {
            "agent": agent,
            "decision": decision,
            "reasoning": reasoning,
            "event_type": "ai_decision"
        }
        if context:
            decision_context.update(context)
        
        self.info("AI_DECISION", f"Agent {agent} made decision: {decision}", decision_context)
    
    def log_crew_creation(self, crew_name: str, task: str, agents: List[str], 
                         context: Dict[str, Any] = None):
        """Log crew creation process."""
        creation_context = {
            "crew_name": crew_name,
            "task": task,
            "agent_count": len(agents),
            "agents": agents,
            "event_type": "crew_creation"
        }
        if context:
            creation_context.update(context)
        
        self.info("CREW_CREATION", f"Created crew '{crew_name}' with {len(agents)} agents", creation_context)
    
    def log_crew_execution(self, crew_name: str, execution_id: str, status: str,
                          context: Dict[str, Any] = None):
        """Log crew execution events."""
        execution_context = {
            "crew_name": crew_name,
            "execution_id": execution_id,
            "status": status,
            "event_type": "crew_execution"
        }
        if context:
            execution_context.update(context)
        
        self.info("CREW_EXECUTION", f"Crew '{crew_name}' execution {status}", execution_context)
    
    def log_tool_usage(self, tool_name: str, agent: str, success: bool, 
                      duration: float = None, context: Dict[str, Any] = None):
        """Log tool usage events."""
        tool_context = {
            "tool_name": tool_name,
            "agent": agent,
            "success": success,
            "duration": duration,
            "event_type": "tool_usage"
        }
        if context:
            tool_context.update(context)
        
        status = "successful" if success else "failed"
        self.info("TOOL_USAGE", f"Tool '{tool_name}' used by {agent} - {status}", tool_context)
    
    def log_llm_call(self, provider: str, model: str, tokens_used: int, cost: float,
                    duration: float = None, context: Dict[str, Any] = None):
        """Log LLM API calls."""
        llm_context = {
            "provider": provider,
            "model": model,
            "tokens_used": tokens_used,
            "cost": cost,
            "duration": duration,
            "event_type": "llm_call"
        }
        if context:
            llm_context.update(context)
        
        self.info("LLM_CALL", f"LLM call: {provider}/{model} - {tokens_used} tokens, ${cost:.4f}", llm_context)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = None,
                              context: Dict[str, Any] = None):
        """Log performance metrics."""
        metric_context = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "event_type": "performance_metric"
        }
        if context:
            metric_context.update(context)
        
        unit_str = f" {unit}" if unit else ""
        self.info("PERFORMANCE", f"Metric: {metric_name} = {value}{unit_str}", metric_context)
    
    def log_error_with_context(self, component: str, error: Exception, 
                              context: Dict[str, Any] = None):
        """Log error with full context and stack trace."""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "event_type": "error"
        }
        if context:
            error_context.update(context)
        
        self.error(component, f"Error: {type(error).__name__}: {str(error)}", 
                  error_context, exc_info=True)
    
    def get_log_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of logs for the specified time period."""
        # This would typically query a log database or parse log files
        # For now, return a basic structure
        return {
            "period_hours": hours,
            "total_entries": 0,
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
            "debug_count": 0,
            "most_active_components": [],
            "error_rate": 0.0
        }
    
    def export_logs(self, output_file: str, hours: int = 24, level: str = "INFO"):
        """Export logs to a file."""
        # This would typically export from a log database
        # For now, create a placeholder
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "period_hours": hours,
            "log_level": level,
            "logs": []
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.info("LOGGER", f"Exported logs to {output_file}", {"export_file": output_file})


# Global logger instance
_global_logger: Optional[AgentForgeLogger] = None


def get_logger(name: str = "agentforge") -> AgentForgeLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = AgentForgeLogger(name)
    return _global_logger


def setup_logging(log_level: LogLevel = LogLevel.INFO, log_file: Optional[str] = None):
    """Setup global logging configuration."""
    global _global_logger
    _global_logger = agentforgeLogger(
        name="agentforge",
        log_level=log_level,
        log_file=log_file,
        enable_console=True
    )
    return _global_logger
