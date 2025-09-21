"""
Database implementation for AgentForge.

This module provides a simple in-memory database implementation
for storing agents, crews, and execution logs.
"""

import json
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from .models import AgentModel, CrewModel, ExecutionResult, ExecutionStatus


class Database:
    """Simple in-memory database for AgentForge."""
    
    def __init__(self, data_dir: str = ".agentforge/data"):
        """Initialize the database."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage
        self.agents: Dict[str, AgentModel] = {}
        self.crews: Dict[str, CrewModel] = {}
        self.executions: Dict[str, ExecutionResult] = {}
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load data from files."""
        try:
            # Load agents
            agents_file = self.data_dir / "agents.json"
            if agents_file.exists():
                with open(agents_file, 'r') as f:
                    agents_data = json.load(f)
                    for agent_data in agents_data:
                        agent = AgentModel(**agent_data)
                        self.agents[agent.id] = agent
            
            # Load crews
            crews_file = self.data_dir / "crews.json"
            if crews_file.exists():
                with open(crews_file, 'r') as f:
                    crews_data = json.load(f)
                    for crew_data in crews_data:
                        # Convert agent data back to AgentModel objects
                        agents = [AgentModel(**agent_data) for agent_data in crew_data.get('agents', [])]
                        crew_data['agents'] = agents
                        crew = CrewModel(**crew_data)
                        self.crews[crew.id] = crew
            
            # Load executions
            executions_file = self.data_dir / "executions.json"
            if executions_file.exists():
                with open(executions_file, 'r') as f:
                    executions_data = json.load(f)
                    for exec_data in executions_data:
                        exec_result = ExecutionResult(**exec_data)
                        self.executions[exec_result.id] = exec_result
        except Exception as e:
            print(f"Warning: Could not load existing data: {e}")
    
    def _save_data(self):
        """Save data to files."""
        try:
            # Save agents
            agents_file = self.data_dir / "agents.json"
            agents_data = []
            for agent in self.agents.values():
                agent_dict = {
                    'id': agent.id,
                    'name': agent.name,
                    'role': agent.role,
                    'goal': agent.goal,
                    'backstory': agent.backstory,
                    'tools': agent.tools,
                    'memory_type': agent.memory_type,
                    'max_iter': agent.max_iter,
                    'allow_delegation': agent.allow_delegation,
                    'created_at': agent.created_at.isoformat() if agent.created_at else None,
                    'updated_at': agent.updated_at.isoformat() if agent.updated_at else None
                }
                agents_data.append(agent_dict)
            
            with open(agents_file, 'w') as f:
                json.dump(agents_data, f, indent=2)
            
            # Save crews
            crews_file = self.data_dir / "crews.json"
            crews_data = []
            for crew in self.crews.values():
                crew_dict = {
                    'id': crew.id,
                    'name': crew.name,
                    'task': crew.task,
                    'description': crew.description,
                    'agents': [
                        {
                            'id': agent.id,
                            'name': agent.name,
                            'role': agent.role,
                            'goal': agent.goal,
                            'backstory': agent.backstory,
                            'tools': agent.tools,
                            'memory_type': agent.memory_type,
                            'max_iter': agent.max_iter,
                            'allow_delegation': agent.allow_delegation,
                            'created_at': agent.created_at.isoformat() if agent.created_at else None,
                            'updated_at': agent.updated_at.isoformat() if agent.updated_at else None
                        }
                        for agent in crew.agents
                    ],
                    'expected_output': crew.expected_output,
                    'complexity': crew.complexity,
                    'estimated_time': crew.estimated_time,
                    'process_type': crew.process_type,
                    'created_at': crew.created_at.isoformat() if crew.created_at else None,
                    'updated_at': crew.updated_at.isoformat() if crew.updated_at else None
                }
                crews_data.append(crew_dict)
            
            with open(crews_file, 'w') as f:
                json.dump(crews_data, f, indent=2)
            
            # Save executions
            executions_file = self.data_dir / "executions.json"
            executions_data = []
            for exec_result in self.executions.values():
                exec_dict = {
                    'id': exec_result.id,
                    'crew_id': exec_result.crew_id,
                    'input_data': exec_result.input_data,
                    'output': exec_result.output,
                    'status': exec_result.status.value,
                    'execution_time': exec_result.execution_time,
                    'cost': exec_result.cost,
                    'quality_score': exec_result.quality_score,
                    'error_message': exec_result.error_message,
                    'logs': exec_result.logs,
                    'created_at': exec_result.created_at.isoformat() if exec_result.created_at else None
                }
                executions_data.append(exec_dict)
            
            with open(executions_file, 'w') as f:
                json.dump(executions_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save data: {e}")


class AgentRepository:
    """Repository for managing agents."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create(self, agent: AgentModel) -> AgentModel:
        """Create a new agent."""
        if not agent.id:
            agent.id = str(uuid.uuid4())
        agent.created_at = datetime.utcnow()
        agent.updated_at = datetime.utcnow()
        self.db.agents[agent.id] = agent
        self.db._save_data()
        return agent
    
    def get_by_id(self, agent_id: str) -> Optional[AgentModel]:
        """Get agent by ID."""
        return self.db.agents.get(agent_id)
    
    def get_by_name(self, name: str) -> Optional[AgentModel]:
        """Get agent by name."""
        for agent in self.db.agents.values():
            if agent.name == name:
                return agent
        return None
    
    def list_all(self) -> List[AgentModel]:
        """List all agents."""
        return list(self.db.agents.values())
    
    def update(self, agent: AgentModel) -> AgentModel:
        """Update an agent."""
        agent.updated_at = datetime.utcnow()
        self.db.agents[agent.id] = agent
        self.db._save_data()
        return agent
    
    def delete(self, agent_id: str) -> bool:
        """Delete an agent."""
        if agent_id in self.db.agents:
            del self.db.agents[agent_id]
            self.db._save_data()
            return True
        return False


class CrewRepository:
    """Repository for managing crews."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create(self, crew: CrewModel) -> CrewModel:
        """Create a new crew."""
        if not crew.id:
            crew.id = str(uuid.uuid4())
        crew.created_at = datetime.utcnow()
        crew.updated_at = datetime.utcnow()
        self.db.crews[crew.id] = crew
        self.db._save_data()
        return crew
    
    def get_by_id(self, crew_id: str) -> Optional[CrewModel]:
        """Get crew by ID."""
        return self.db.crews.get(crew_id)
    
    def get_by_name(self, name: str) -> Optional[CrewModel]:
        """Get crew by name."""
        for crew in self.db.crews.values():
            if crew.name == name:
                return crew
        return None
    
    def list_all(self) -> List[CrewModel]:
        """List all crews."""
        return list(self.db.crews.values())
    
    def update(self, crew: CrewModel) -> CrewModel:
        """Update a crew."""
        crew.updated_at = datetime.utcnow()
        self.db.crews[crew.id] = crew
        self.db._save_data()
        return crew
    
    def delete(self, crew_id: str) -> bool:
        """Delete a crew."""
        if crew_id in self.db.crews:
            del self.db.crews[crew_id]
            self.db._save_data()
            return True
        return False


class ExecutionLogRepository:
    """Repository for managing execution logs."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create(self, execution: ExecutionResult) -> ExecutionResult:
        """Create a new execution log."""
        if not execution.id:
            execution.id = str(uuid.uuid4())
        execution.created_at = datetime.utcnow()
        self.db.executions[execution.id] = execution
        self.db._save_data()
        return execution
    
    def get_by_id(self, execution_id: str) -> Optional[ExecutionResult]:
        """Get execution by ID."""
        return self.db.executions.get(execution_id)
    
    def get_by_crew_id(self, crew_id: str) -> List[ExecutionResult]:
        """Get executions by crew ID."""
        return [exec for exec in self.db.executions.values() if exec.crew_id == crew_id]
    
    def list_all(self) -> List[ExecutionResult]:
        """List all executions."""
        return list(self.db.executions.values())
    
    def update(self, execution: ExecutionResult) -> ExecutionResult:
        """Update an execution log."""
        self.db.executions[execution.id] = execution
        self.db._save_data()
        return execution
