"""
Performance tracking and analytics for crew execution.
"""

import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid


@dataclass
class ExecutionMetrics:
    """Metrics for a single crew execution."""
    execution_id: str
    crew_name: str
    task_description: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    
    # Agent metrics
    agent_count: int = 0
    agent_execution_times: Dict[str, float] = None
    agent_success_rates: Dict[str, bool] = None
    
    # Tool metrics
    tool_usage_count: Dict[str, int] = None
    tool_success_rates: Dict[str, float] = None
    tool_execution_times: Dict[str, float] = None
    
    # LLM metrics
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    total_tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_cost: float = 0.0
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    iterations_completed: int = 0
    max_iterations: int = 0
    
    # Quality metrics
    output_quality_score: float = 0.0
    task_completion_percentage: float = 0.0
    user_satisfaction_score: Optional[float] = None
    
    def __post_init__(self):
        if self.agent_execution_times is None:
            self.agent_execution_times = {}
        if self.agent_success_rates is None:
            self.agent_success_rates = {}
        if self.tool_usage_count is None:
            self.tool_usage_count = {}
        if self.tool_success_rates is None:
            self.tool_success_rates = {}
        if self.tool_execution_times is None:
            self.tool_execution_times = {}


class PerformanceTracker:
    """Tracks and analyzes crew execution performance."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path.home() / ".agentforge" / "analytics.db")
        self._ensure_db_directory()
        self._init_database()
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize the analytics database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_metrics (
                    execution_id TEXT PRIMARY KEY,
                    crew_name TEXT NOT NULL,
                    task_description TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_seconds REAL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    agent_count INTEGER,
                    agent_execution_times TEXT,
                    agent_success_rates TEXT,
                    tool_usage_count TEXT,
                    tool_success_rates TEXT,
                    tool_execution_times TEXT,
                    llm_provider TEXT,
                    llm_model TEXT,
                    total_tokens_used INTEGER,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    llm_cost REAL,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL,
                    iterations_completed INTEGER,
                    max_iterations INTEGER,
                    output_quality_score REAL,
                    task_completion_percentage REAL,
                    user_satisfaction_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS crew_performance_history (
                    crew_name TEXT PRIMARY KEY,
                    total_executions INTEGER DEFAULT 0,
                    successful_executions INTEGER DEFAULT 0,
                    average_duration REAL,
                    average_cost REAL,
                    success_rate REAL,
                    last_execution TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crew_name TEXT NOT NULL,
                    recommendation_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    impact_score REAL,
                    implementation_effort TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def start_execution(self, crew_name: str, task_description: str) -> str:
        """Start tracking a new execution."""
        execution_id = str(uuid.uuid4())
        metrics = ExecutionMetrics(
            execution_id=execution_id,
            crew_name=crew_name,
            task_description=task_description,
            start_time=datetime.now()
        )
        
        # Store initial metrics
        self._store_metrics(metrics)
        return execution_id
    
    def update_execution(self, execution_id: str, updates: Dict[str, Any]):
        """Update execution metrics."""
        with sqlite3.connect(self.db_path) as conn:
            # Get current metrics
            cursor = conn.execute(
                "SELECT * FROM execution_metrics WHERE execution_id = ?", 
                (execution_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return
            
            # Convert to ExecutionMetrics object
            metrics = self._row_to_metrics(row)
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            # Update end time and duration if not set
            if metrics.end_time is None and 'end_time' in updates:
                metrics.end_time = updates['end_time']
                if metrics.start_time:
                    duration = (metrics.end_time - metrics.start_time).total_seconds()
                    metrics.duration_seconds = duration
            
            # Store updated metrics
            self._store_metrics(metrics)
    
    def end_execution(self, execution_id: str, success: bool = True, error_message: Optional[str] = None):
        """End execution tracking."""
        end_time = datetime.now()
        updates = {
            'end_time': end_time.isoformat(),
            'success': success,
            'error_message': error_message
        }
        
        self.update_execution(execution_id, updates)
        
        # Update crew performance history
        self._update_crew_history(execution_id)
    
    def get_execution_metrics(self, execution_id: str) -> Optional[ExecutionMetrics]:
        """Get metrics for a specific execution."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM execution_metrics WHERE execution_id = ?", 
                (execution_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_metrics(row)
            return None
    
    def get_crew_performance(self, crew_name: str, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics for a specific crew."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_executions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_executions,
                    AVG(duration_seconds) as avg_duration,
                    AVG(llm_cost) as avg_cost,
                    AVG(output_quality_score) as avg_quality,
                    AVG(task_completion_percentage) as avg_completion
                FROM execution_metrics 
                WHERE crew_name = ? AND start_time >= ?
            """, (crew_name, cutoff_date.isoformat()))
            
            row = cursor.fetchone()
            
            if row and row[0] > 0:
                total, successful, avg_duration, avg_cost, avg_quality, avg_completion = row
                return {
                    'crew_name': crew_name,
                    'total_executions': total,
                    'successful_executions': successful,
                    'success_rate': successful / total if total > 0 else 0,
                    'average_duration_seconds': avg_duration or 0,
                    'average_cost': avg_cost or 0,
                    'average_quality_score': avg_quality or 0,
                    'average_completion_percentage': avg_completion or 0,
                    'period_days': days
                }
            else:
                return {
                    'crew_name': crew_name,
                    'total_executions': 0,
                    'successful_executions': 0,
                    'success_rate': 0,
                    'average_duration_seconds': 0,
                    'average_cost': 0,
                    'average_quality_score': 0,
                    'average_completion_percentage': 0,
                    'period_days': days
                }
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get overall performance summary."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall metrics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_executions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_executions,
                    AVG(duration_seconds) as avg_duration,
                    SUM(llm_cost) as total_cost,
                    AVG(llm_cost) as avg_cost,
                    COUNT(DISTINCT crew_name) as unique_crews
                FROM execution_metrics 
                WHERE start_time >= ?
            """, (cutoff_date.isoformat(),))
            
            overall_row = cursor.fetchone()
            
            # Top performing crews
            cursor = conn.execute("""
                SELECT 
                    crew_name,
                    COUNT(*) as executions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    AVG(duration_seconds) as avg_duration,
                    AVG(output_quality_score) as avg_quality
                FROM execution_metrics 
                WHERE start_time >= ?
                GROUP BY crew_name
                ORDER BY (SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) DESC
                LIMIT 5
            """, (cutoff_date.isoformat(),))
            
            top_crews = []
            for row in cursor.fetchall():
                crew_name, executions, successful, avg_duration, avg_quality = row
                top_crews.append({
                    'crew_name': crew_name,
                    'executions': executions,
                    'success_rate': successful / executions if executions > 0 else 0,
                    'average_duration': avg_duration or 0,
                    'average_quality': avg_quality or 0
                })
            
            # Tool usage statistics
            cursor = conn.execute("""
                SELECT 
                    tool_name,
                    SUM(usage_count) as total_usage,
                    AVG(success_rate) as avg_success_rate
                FROM (
                    SELECT 
                        json_extract(value, '$.tool') as tool_name,
                        json_extract(value, '$.count') as usage_count,
                        json_extract(value, '$.success_rate') as success_rate
                    FROM execution_metrics,
                    json_each(tool_usage_count)
                )
                WHERE start_time >= ?
                GROUP BY tool_name
                ORDER BY total_usage DESC
                LIMIT 10
            """, (cutoff_date.isoformat(),))
            
            tool_usage = []
            for row in cursor.fetchall():
                tool_name, total_usage, avg_success_rate = row
                tool_usage.append({
                    'tool_name': tool_name,
                    'total_usage': total_usage,
                    'average_success_rate': avg_success_rate or 0
                })
            
            if overall_row:
                total, successful, avg_duration, total_cost, avg_cost, unique_crews = overall_row
                return {
                    'period_days': days,
                    'total_executions': total,
                    'successful_executions': successful,
                    'overall_success_rate': successful / total if total > 0 else 0,
                    'average_duration_seconds': avg_duration or 0,
                    'total_cost': total_cost or 0,
                    'average_cost_per_execution': avg_cost or 0,
                    'unique_crews': unique_crews,
                    'top_performing_crews': top_crews,
                    'most_used_tools': tool_usage
                }
            else:
                return {
                    'period_days': days,
                    'total_executions': 0,
                    'successful_executions': 0,
                    'overall_success_rate': 0,
                    'average_duration_seconds': 0,
                    'total_cost': 0,
                    'average_cost_per_execution': 0,
                    'unique_crews': 0,
                    'top_performing_crews': [],
                    'most_used_tools': []
                }
    
    def get_optimization_recommendations(self, crew_name: str) -> List[Dict[str, Any]]:
        """Get optimization recommendations for a crew."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    recommendation_type,
                    title,
                    description,
                    impact_score,
                    implementation_effort,
                    created_at
                FROM optimization_recommendations 
                WHERE crew_name = ?
                ORDER BY impact_score DESC
            """, (crew_name,))
            
            recommendations = []
            for row in cursor.fetchall():
                rec_type, title, description, impact_score, effort, created_at = row
                recommendations.append({
                    'type': rec_type,
                    'title': title,
                    'description': description,
                    'impact_score': impact_score or 0,
                    'implementation_effort': effort or 'medium',
                    'created_at': created_at
                })
            
            return recommendations
    
    def add_optimization_recommendation(self, crew_name: str, recommendation_type: str, 
                                      title: str, description: str, impact_score: float = 0.0,
                                      implementation_effort: str = 'medium'):
        """Add an optimization recommendation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO optimization_recommendations 
                (crew_name, recommendation_type, title, description, impact_score, implementation_effort)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (crew_name, recommendation_type, title, description, impact_score, implementation_effort))
    
    def _store_metrics(self, metrics: ExecutionMetrics):
        """Store metrics in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO execution_metrics (
                    execution_id, crew_name, task_description, start_time, end_time,
                    duration_seconds, success, error_message, agent_count,
                    agent_execution_times, agent_success_rates, tool_usage_count,
                    tool_success_rates, tool_execution_times, llm_provider, llm_model,
                    total_tokens_used, prompt_tokens, completion_tokens, llm_cost,
                    memory_usage_mb, cpu_usage_percent, iterations_completed,
                    max_iterations, output_quality_score, task_completion_percentage,
                    user_satisfaction_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.execution_id, metrics.crew_name, metrics.task_description,
                metrics.start_time.isoformat(), metrics.end_time.isoformat() if metrics.end_time else None,
                metrics.duration_seconds, metrics.success, metrics.error_message, metrics.agent_count,
                json.dumps(metrics.agent_execution_times), json.dumps(metrics.agent_success_rates),
                json.dumps(metrics.tool_usage_count), json.dumps(metrics.tool_success_rates),
                json.dumps(metrics.tool_execution_times), metrics.llm_provider, metrics.llm_model,
                metrics.total_tokens_used, metrics.prompt_tokens, metrics.completion_tokens,
                metrics.llm_cost, metrics.memory_usage_mb, metrics.cpu_usage_percent,
                metrics.iterations_completed, metrics.max_iterations, metrics.output_quality_score,
                metrics.task_completion_percentage, metrics.user_satisfaction_score
            ))
    
    def _row_to_metrics(self, row) -> ExecutionMetrics:
        """Convert database row to ExecutionMetrics object."""
        return ExecutionMetrics(
            execution_id=row[0],
            crew_name=row[1],
            task_description=row[2],
            start_time=datetime.fromisoformat(row[3]),
            end_time=datetime.fromisoformat(row[4]) if row[4] else None,
            duration_seconds=row[5],
            success=bool(row[6]),
            error_message=row[7],
            agent_count=row[8] or 0,
            agent_execution_times=json.loads(row[9]) if row[9] else {},
            agent_success_rates=json.loads(row[10]) if row[10] else {},
            tool_usage_count=json.loads(row[11]) if row[11] else {},
            tool_success_rates=json.loads(row[12]) if row[12] else {},
            tool_execution_times=json.loads(row[13]) if row[13] else {},
            llm_provider=row[14],
            llm_model=row[15],
            total_tokens_used=row[16] or 0,
            prompt_tokens=row[17] or 0,
            completion_tokens=row[18] or 0,
            llm_cost=row[19] or 0.0,
            memory_usage_mb=row[20] or 0.0,
            cpu_usage_percent=row[21] or 0.0,
            iterations_completed=row[22] or 0,
            max_iterations=row[23] or 0,
            output_quality_score=row[24] or 0.0,
            task_completion_percentage=row[25] or 0.0,
            user_satisfaction_score=row[26]
        )
    
    def _update_crew_history(self, execution_id: str):
        """Update crew performance history."""
        metrics = self.get_execution_metrics(execution_id)
        if not metrics:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Get current history
            cursor = conn.execute(
                "SELECT * FROM crew_performance_history WHERE crew_name = ?",
                (metrics.crew_name,)
            )
            row = cursor.fetchone()
            
            if row:
                # Update existing history
                total_executions = row[1] + 1
                successful_executions = row[2] + (1 if metrics.success else 0)
                success_rate = successful_executions / total_executions
                
                # Calculate new averages
                cursor = conn.execute("""
                    SELECT 
                        AVG(duration_seconds) as avg_duration,
                        AVG(llm_cost) as avg_cost
                    FROM execution_metrics 
                    WHERE crew_name = ?
                """, (metrics.crew_name,))
                avg_row = cursor.fetchone()
                avg_duration = avg_row[0] if avg_row[0] else 0
                avg_cost = avg_row[1] if avg_row[1] else 0
                
                conn.execute("""
                    UPDATE crew_performance_history 
                    SET total_executions = ?, successful_executions = ?, success_rate = ?,
                        average_duration = ?, average_cost = ?, last_execution = ?, updated_at = ?
                    WHERE crew_name = ?
                """, (total_executions, successful_executions, success_rate, avg_duration, avg_cost,
                      metrics.end_time.isoformat() if metrics.end_time else None, datetime.now().isoformat(),
                      metrics.crew_name))
            else:
                # Create new history
                conn.execute("""
                    INSERT INTO crew_performance_history 
                    (crew_name, total_executions, successful_executions, success_rate, 
                     average_duration, average_cost, last_execution)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (metrics.crew_name, 1, 1 if metrics.success else 0, 
                      1.0 if metrics.success else 0.0, metrics.duration_seconds or 0,
                      metrics.llm_cost, metrics.end_time.isoformat() if metrics.end_time else None))
