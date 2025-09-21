"""
Adaptive Agent Manager for AgentForge.

This module implements a reinforcement learning-based system that can dynamically
create, modify, and optimize agents based on performance feedback and task requirements.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path

from .config import Config
from .llm_provider import get_llm_config_for_crewai
from .master_agent import MasterAgent
from .crew_designer import CrewDesigner
from .file_generator import CrewFileGenerator
from ..analytics.performance_tracker import PerformanceTracker
from ..logging.logger import get_logger

class AgentCreationTrigger(Enum):
    """Triggers for creating new agents."""
    PERFORMANCE_DROP = "performance_drop"
    NEW_CAPABILITY_NEEDED = "new_capability_needed"
    TASK_COMPLEXITY_INCREASE = "task_complexity_increase"
    SPECIALIZATION_REQUIRED = "specialization_required"
    FAILURE_RECOVERY = "failure_recovery"

@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an agent."""
    agent_name: str
    success_rate: float
    avg_execution_time: float
    cost_efficiency: float
    quality_score: float
    task_complexity: float
    specialization_level: float
    last_updated: float

@dataclass
class AgentCreationDecision:
    """Decision to create a new agent."""
    trigger: AgentCreationTrigger
    reasoning: str
    new_agent_spec: Dict[str, Any]
    confidence: float
    expected_improvement: float
    creation_cost: float

class AdaptiveAgentManager:
    """Manages adaptive agent creation using reinforcement learning principles."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("adaptive_agent_manager")
        self.performance_tracker = PerformanceTracker()
        
        # RL parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.3
        self.discount_factor = 0.95
        
        # Agent performance tracking
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.creation_history: List[AgentCreationDecision] = []
        
        # Performance thresholds
        self.performance_thresholds = {
            "min_success_rate": 0.7,
            "max_execution_time": 300,  # seconds
            "min_quality_score": 0.6,
            "max_cost_per_task": 0.50  # dollars
        }
        
        # Specialization patterns
        self.specialization_patterns = {
            "research": ["web_search", "data_analysis", "report_generation"],
            "creative": ["content_generation", "design", "storytelling"],
            "technical": ["code_generation", "debugging", "system_analysis"],
            "analytical": ["data_processing", "statistical_analysis", "pattern_recognition"],
            "communication": ["writing", "presentation", "translation"]
        }
    
    def analyze_agent_performance(self, crew_name: str, days: int = 7) -> Dict[str, Any]:
        """Analyze agent performance and identify improvement opportunities."""
        try:
            # Get performance data
            performance_data = self.performance_tracker.get_crew_performance(crew_name, days)
            
            if not performance_data:
                return {"status": "no_data", "recommendations": []}
            
            # Calculate metrics for each agent
            agent_analysis = {}
            for agent_name in performance_data.get("agents", []):
                metrics = self._calculate_agent_metrics(agent_name, performance_data)
                agent_analysis[agent_name] = metrics
                
                # Update stored metrics
                self.agent_metrics[agent_name] = metrics
            
            # Identify improvement opportunities
            recommendations = self._identify_improvement_opportunities(agent_analysis)
            
            return {
                "status": "success",
                "agent_analysis": agent_analysis,
                "recommendations": recommendations,
                "overall_performance": self._calculate_overall_performance(agent_analysis)
            }
            
        except Exception as e:
            self.logger.error("ADAPTIVE_ANALYSIS", f"Error analyzing agent performance: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_agent_metrics(self, agent_name: str, performance_data: Dict) -> AgentPerformanceMetrics:
        """Calculate performance metrics for a specific agent."""
        # This would integrate with your performance tracking system
        # For now, using mock data structure
        return AgentPerformanceMetrics(
            agent_name=agent_name,
            success_rate=0.85,  # Would be calculated from actual data
            avg_execution_time=120.0,
            cost_efficiency=0.75,
            quality_score=0.8,
            task_complexity=0.6,
            specialization_level=0.5,
            last_updated=time.time()
        )
    
    def _identify_improvement_opportunities(self, agent_analysis: Dict) -> List[Dict]:
        """Identify opportunities for agent improvement or creation."""
        recommendations = []
        
        for agent_name, metrics in agent_analysis.items():
            # Check performance thresholds
            if metrics.success_rate < self.performance_thresholds["min_success_rate"]:
                recommendations.append({
                    "type": "performance_improvement",
                    "agent": agent_name,
                    "issue": "low_success_rate",
                    "current_value": metrics.success_rate,
                    "threshold": self.performance_thresholds["min_success_rate"],
                    "suggestion": "Create specialized agent for failed task types"
                })
            
            if metrics.avg_execution_time > self.performance_thresholds["max_execution_time"]:
                recommendations.append({
                    "type": "performance_improvement",
                    "agent": agent_name,
                    "issue": "slow_execution",
                    "current_value": metrics.avg_execution_time,
                    "threshold": self.performance_thresholds["max_execution_time"],
                    "suggestion": "Create optimized agent for faster execution"
                })
            
            if metrics.quality_score < self.performance_thresholds["min_quality_score"]:
                recommendations.append({
                    "type": "quality_improvement",
                    "agent": agent_name,
                    "issue": "low_quality",
                    "current_value": metrics.quality_score,
                    "threshold": self.performance_thresholds["min_quality_score"],
                    "suggestion": "Create quality-focused specialist agent"
                })
        
        return recommendations
    
    def decide_agent_creation(self, crew_name: str, task_context: Dict[str, Any]) -> Optional[AgentCreationDecision]:
        """Decide whether to create a new agent based on current performance and task needs."""
        try:
            # Analyze current performance
            performance_analysis = self.analyze_agent_performance(crew_name)
            
            if performance_analysis["status"] != "success":
                return None
            
            # Check for creation triggers
            triggers = self._identify_creation_triggers(performance_analysis, task_context)
            
            if not triggers:
                return None
            
            # Use RL to decide on agent creation
            decision = self._rl_agent_creation_decision(triggers, task_context)
            
            if decision:
                self.creation_history.append(decision)
                self.logger.info("ADAPTIVE_DECISION", f"Agent creation decision made: {decision.reasoning}")
            
            return decision
            
        except Exception as e:
            self.logger.error("ADAPTIVE_DECISION", f"Error in agent creation decision: {str(e)}")
            return None
    
    def _identify_creation_triggers(self, performance_analysis: Dict, task_context: Dict) -> List[AgentCreationTrigger]:
        """Identify triggers for agent creation."""
        triggers = []
        
        # Check performance-based triggers
        for recommendation in performance_analysis.get("recommendations", []):
            if recommendation["type"] == "performance_improvement":
                if recommendation["issue"] == "low_success_rate":
                    triggers.append(AgentCreationTrigger.PERFORMANCE_DROP)
                elif recommendation["issue"] == "slow_execution":
                    triggers.append(AgentCreationTrigger.SPECIALIZATION_REQUIRED)
                elif recommendation["issue"] == "low_quality":
                    triggers.append(AgentCreationTrigger.SPECIALIZATION_REQUIRED)
        
        # Check task complexity
        task_complexity = task_context.get("complexity", 0.5)
        if task_complexity > 0.8:
            triggers.append(AgentCreationTrigger.TASK_COMPLEXITY_INCREASE)
        
        # Check for new capabilities needed
        required_capabilities = task_context.get("required_capabilities", [])
        current_capabilities = self._get_current_capabilities(performance_analysis)
        missing_capabilities = set(required_capabilities) - set(current_capabilities)
        
        if missing_capabilities:
            triggers.append(AgentCreationTrigger.NEW_CAPABILITY_NEEDED)
        
        return triggers
    
    def _rl_agent_creation_decision(self, triggers: List[AgentCreationTrigger], task_context: Dict) -> Optional[AgentCreationDecision]:
        """Use reinforcement learning to decide on agent creation."""
        # Calculate expected value of creating an agent
        expected_value = self._calculate_expected_value(triggers, task_context)
        
        # Apply exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Exploration: create agent with some probability
            if expected_value > 0.3:  # Threshold for exploration
                return self._create_agent_decision(triggers, task_context, expected_value)
        else:
            # Exploitation: create agent if expected value is high enough
            if expected_value > 0.7:  # Higher threshold for exploitation
                return self._create_agent_decision(triggers, task_context, expected_value)
        
        return None
    
    def _calculate_expected_value(self, triggers: List[AgentCreationTrigger], task_context: Dict) -> float:
        """Calculate expected value of creating a new agent."""
        base_value = 0.0
        
        # Weight different triggers
        trigger_weights = {
            AgentCreationTrigger.PERFORMANCE_DROP: 0.8,
            AgentCreationTrigger.NEW_CAPABILITY_NEEDED: 0.9,
            AgentCreationTrigger.TASK_COMPLEXITY_INCREASE: 0.6,
            AgentCreationTrigger.SPECIALIZATION_REQUIRED: 0.7,
            AgentCreationTrigger.FAILURE_RECOVERY: 0.9
        }
        
        for trigger in triggers:
            base_value += trigger_weights.get(trigger, 0.5)
        
        # Normalize by number of triggers
        if triggers:
            base_value /= len(triggers)
        
        # Adjust based on historical performance
        historical_success = self._get_historical_creation_success()
        base_value *= historical_success
        
        return min(base_value, 1.0)  # Cap at 1.0
    
    def _create_agent_decision(self, triggers: List[AgentCreationTrigger], task_context: Dict, expected_value: float) -> AgentCreationDecision:
        """Create a decision to create a new agent."""
        # Determine agent specialization based on triggers
        specialization = self._determine_specialization(triggers, task_context)
        
        # Generate agent specification
        agent_spec = self._generate_agent_specification(specialization, task_context)
        
        # Calculate expected improvement and cost
        expected_improvement = expected_value * 0.3  # Assume 30% improvement
        creation_cost = self._estimate_creation_cost(agent_spec)
        
        return AgentCreationDecision(
            trigger=triggers[0],  # Primary trigger
            reasoning=f"Creating specialized agent for {specialization} based on {len(triggers)} triggers",
            new_agent_spec=agent_spec,
            confidence=expected_value,
            expected_improvement=expected_improvement,
            creation_cost=creation_cost
        )
    
    def _determine_specialization(self, triggers: List[AgentCreationTrigger], task_context: Dict) -> str:
        """Determine the specialization for the new agent."""
        # Map triggers to specializations
        trigger_specialization_map = {
            AgentCreationTrigger.PERFORMANCE_DROP: "optimization",
            AgentCreationTrigger.NEW_CAPABILITY_NEEDED: "capability",
            AgentCreationTrigger.TASK_COMPLEXITY_INCREASE: "complexity",
            AgentCreationTrigger.SPECIALIZATION_REQUIRED: "specialization",
            AgentCreationTrigger.FAILURE_RECOVERY: "recovery"
        }
        
        # Get primary specialization
        primary_trigger = triggers[0]
        base_specialization = trigger_specialization_map.get(primary_trigger, "general")
        
        # Refine based on task context
        task_domain = task_context.get("domain", "general")
        if task_domain in self.specialization_patterns:
            return f"{base_specialization}_{task_domain}"
        
        return base_specialization
    
    def _generate_agent_specification(self, specialization: str, task_context: Dict) -> Dict[str, Any]:
        """Generate specification for the new agent."""
        # This would integrate with your existing agent generation system
        return {
            "name": f"adaptive_{specialization}_{int(time.time())}",
            "role": f"Specialized {specialization} agent",
            "goal": f"Optimize performance for {specialization} tasks",
            "backstory": f"AI agent specialized in {specialization} created through adaptive learning",
            "tools": self._get_specialized_tools(specialization),
            "memory_type": "long_term",
            "max_iter": 10,
            "allow_delegation": True,
            "specialization": specialization,
            "adaptive": True
        }
    
    def _get_specialized_tools(self, specialization: str) -> List[str]:
        """Get tools appropriate for the specialization."""
        tool_mapping = {
            "optimization": ["performance_analyzer", "optimizer", "benchmark"],
            "capability": ["capability_analyzer", "skill_assessor", "learning_tool"],
            "complexity": ["complexity_analyzer", "task_breaker", "coordination_tool"],
            "specialization": ["domain_expert", "specialist_tool", "expert_system"],
            "recovery": ["error_analyzer", "recovery_tool", "fallback_system"]
        }
        
        return tool_mapping.get(specialization, ["general_tool", "analyzer"])
    
    def create_adaptive_agent(self, decision: AgentCreationDecision) -> Dict[str, Any]:
        """Create the adaptive agent based on the decision."""
        try:
            self.logger.info("ADAPTIVE_AGENT", f"Creating adaptive agent: {decision.new_agent_spec['name']}")
            
            # Use existing agent creation system
            master_agent = MasterAgent(self.config)
            crew_designer = CrewDesigner(self.config)
            
            # Create a minimal crew with the new agent
            crew_spec = {
                "name": f"adaptive_crew_{int(time.time())}",
                "description": f"Adaptive crew with {decision.new_agent_spec['name']}",
                "agents": [decision.new_agent_spec],
                "tasks": [{
                    "name": f"adaptive_task_{specialization}",
                    "description": f"Task for {decision.new_agent_spec['role']}",
                    "expected_output": f"Optimized output from {decision.new_agent_spec['name']}",
                    "agent_name": decision.new_agent_spec['name']
                }],
                "process_type": "sequential"
            }
            
            # Generate the crew files
            file_generator = CrewFileGenerator()
            crew_path = file_generator.generate_crew_project(crew_spec)
            
            # Track the creation
            self._track_agent_creation(decision, crew_path)
            
            return {
                "status": "success",
                "agent_name": decision.new_agent_spec['name'],
                "crew_path": crew_path,
                "specialization": decision.new_agent_spec['specialization'],
                "expected_improvement": decision.expected_improvement
            }
            
        except Exception as e:
            self.logger.error("ADAPTIVE_AGENT", f"Error creating adaptive agent: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _track_agent_creation(self, decision: AgentCreationDecision, crew_path: str):
        """Track the creation of an adaptive agent."""
        creation_record = {
            "timestamp": time.time(),
            "decision": decision,
            "crew_path": crew_path,
            "status": "created"
        }
        
        # Store in performance tracker or separate database
        # This would integrate with your analytics system
        self.logger.info("ADAPTIVE_TRACKING", f"Adaptive agent creation tracked: {creation_record}")
    
    def _get_current_capabilities(self, performance_analysis: Dict) -> List[str]:
        """Get current capabilities of the crew."""
        # This would analyze the current agents and their tools
        # For now, return mock capabilities
        return ["general_analysis", "content_generation", "data_processing"]
    
    def _get_historical_creation_success(self) -> float:
        """Get historical success rate of agent creation decisions."""
        if not self.creation_history:
            return 0.5  # Default confidence
        
        # Calculate success rate from historical data
        # This would integrate with your performance tracking
        return 0.75  # Mock success rate
    
    def _estimate_creation_cost(self, agent_spec: Dict) -> float:
        """Estimate the cost of creating the agent."""
        # Base cost for agent creation
        base_cost = 0.10  # $0.10
        
        # Additional cost based on complexity
        complexity_multiplier = 1.0 + (len(agent_spec.get("tools", [])) * 0.1)
        
        return base_cost * complexity_multiplier
    
    def _calculate_overall_performance(self, agent_analysis: Dict) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        if not agent_analysis:
            return {"overall_score": 0.0}
        
        # Calculate weighted average of all metrics
        total_score = 0.0
        count = 0
        
        for agent_name, metrics in agent_analysis.items():
            agent_score = (
                metrics.success_rate * 0.3 +
                (1.0 - min(metrics.avg_execution_time / 300, 1.0)) * 0.2 +
                metrics.quality_score * 0.3 +
                metrics.cost_efficiency * 0.2
            )
            total_score += agent_score
            count += 1
        
        return {
            "overall_score": total_score / count if count > 0 else 0.0,
            "agent_count": count
        }
    
    def update_learning_parameters(self, feedback: Dict[str, Any]):
        """Update RL parameters based on feedback."""
        # Update learning rate based on performance
        if feedback.get("performance_improved", False):
            self.learning_rate = min(self.learning_rate * 1.1, 0.5)
            self.exploration_rate = max(self.exploration_rate * 0.9, 0.1)
        else:
            self.learning_rate = max(self.learning_rate * 0.9, 0.01)
            self.exploration_rate = min(self.exploration_rate * 1.1, 0.5)
        
        self.logger.info(f"Updated learning parameters: lr={self.learning_rate:.3f}, exploration={self.exploration_rate:.3f}")
    
    def get_adaptive_insights(self) -> Dict[str, Any]:
        """Get insights about the adaptive system."""
        return {
            "total_agents_created": len(self.creation_history),
            "current_learning_rate": self.learning_rate,
            "current_exploration_rate": self.exploration_rate,
            "average_confidence": np.mean([d.confidence for d in self.creation_history]) if self.creation_history else 0.0,
            "recent_triggers": [d.trigger.value for d in self.creation_history[-5:]],
            "performance_trend": self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend over time."""
        # This would analyze historical performance data
        # For now, return mock trend
        return "improving"
