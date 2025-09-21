"""
Database models for AgentForge.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class ExecutionStatus(Enum):
    """Execution status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentModel:
    """Agent model for database storage."""
    id: str
    name: str
    role: str
    goal: str
    backstory: str
    tools: List[str]
    memory_type: str = "short_term"
    max_iter: int = 5
    allow_delegation: bool = False
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class CrewModel:
    """Crew model for database storage."""
    id: str
    name: str
    task: str
    description: str
    agents: List[AgentModel]
    expected_output: str
    complexity: str = "moderate"
    estimated_time: int = 15
    process_type: str = "sequential"
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class ExecutionResult:
    """Execution result model for database storage."""
    id: str
    crew_id: str
    input_data: str
    output: str
    status: ExecutionStatus
    execution_time: int
    cost: float = 0.0
    quality_score: float = 0.0
    error_message: Optional[str] = None
    logs: List[Dict[str, Any]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.logs is None:
            self.logs = []
