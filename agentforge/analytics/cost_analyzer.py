"""
Cost analysis and estimation for crew execution.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class CostEstimate:
    """Cost estimate for crew execution."""
    crew_name: str
    estimated_total_cost: float
    cost_breakdown: Dict[str, float]
    cost_per_agent: Dict[str, float]
    cost_per_tool: Dict[str, float]
    estimated_tokens: int
    estimated_duration_minutes: float
    confidence_score: float  # 0-1, how confident we are in this estimate
    created_at: datetime


class CostAnalyzer:
    """Analyzes and estimates costs for crew execution."""
    
    # Pricing data (as of 2024, in USD per 1K tokens)
    PRICING_DATA = {
        "openai": {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        },
        "anthropic": {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        },
        "google": {
            "gemini-pro": {"input": 0.0005, "output": 0.0015},
            "gemini-pro-vision": {"input": 0.0005, "output": 0.0015},
        },
        "deepseek": {
            "deepseek-chat": {"input": 0.0014, "output": 0.0028},
            "deepseek-coder": {"input": 0.0014, "output": 0.0028},
        },
        "ollama": {
            "llama3.1": {"input": 0.0, "output": 0.0},  # Local, no cost
            "llama3.1:8b": {"input": 0.0, "output": 0.0},
            "llama3.1:70b": {"input": 0.0, "output": 0.0},
        },
        "llamacpp": {
            "llama-3.1-8b": {"input": 0.0, "output": 0.0},  # Local, no cost
            "llama-3.1-70b": {"input": 0.0, "output": 0.0},
        }
    }
    
    # Tool usage costs (per call)
    TOOL_COSTS = {
        "WebSearchTool": 0.001,
        "ScrapeWebsiteTool": 0.002,
        "GithubSearchTool": 0.001,
        "CodeInterpreterTool": 0.005,
        "FileReadTool": 0.0001,
        "FileWriteTool": 0.0001,
        "CSVSearchTool": 0.0005,
        "DirectoryReadTool": 0.0001,
        "DirectorySearchTool": 0.0002,
        "EmailTool": 0.01,
        "SlackTool": 0.005,
        "TelegramTool": 0.005,
    }
    
    def __init__(self, performance_tracker=None):
        self.performance_tracker = performance_tracker
    
    def estimate_crew_cost(self, crew_spec: Any, llm_provider: str = "openai", 
                          llm_model: str = "gpt-4") -> CostEstimate:
        """Estimate the cost of executing a crew."""
        # Get pricing for the specified model
        provider_pricing = self.PRICING_DATA.get(llm_provider, {})
        model_pricing = provider_pricing.get(llm_model, {"input": 0.01, "output": 0.02})
        
        # Estimate tokens based on crew complexity
        estimated_tokens = self._estimate_tokens(crew_spec)
        
        # Calculate LLM costs
        input_tokens = int(estimated_tokens * 0.7)  # Assume 70% input, 30% output
        output_tokens = int(estimated_tokens * 0.3)
        
        llm_cost = (input_tokens / 1000 * model_pricing["input"] + 
                   output_tokens / 1000 * model_pricing["output"])
        
        # Calculate tool costs
        tool_costs = self._calculate_tool_costs(crew_spec)
        
        # Calculate agent costs (based on iterations)
        agent_costs = self._calculate_agent_costs(crew_spec, llm_cost)
        
        # Total cost
        total_cost = llm_cost + tool_costs + agent_costs
        
        # Cost breakdown
        cost_breakdown = {
            "llm_cost": llm_cost,
            "tool_costs": tool_costs,
            "agent_costs": agent_costs,
            "total": total_cost
        }
        
        # Per-agent costs
        cost_per_agent = {}
        if hasattr(crew_spec, 'agents'):
            for agent in crew_spec.agents:
                agent_name = getattr(agent, 'name', 'unknown')
                # Distribute LLM cost evenly among agents
                agent_llm_cost = llm_cost / len(crew_spec.agents) if crew_spec.agents else 0
                agent_tool_cost = self._calculate_agent_tool_costs(agent)
                cost_per_agent[agent_name] = agent_llm_cost + agent_tool_cost
        
        # Per-tool costs
        cost_per_tool = self._calculate_per_tool_costs(crew_spec)
        
        # Estimate duration
        estimated_duration = self._estimate_duration(crew_spec)
        
        # Confidence score based on historical data
        confidence = self._calculate_confidence_score(crew_spec, llm_provider, llm_model)
        
        return CostEstimate(
            crew_name=getattr(crew_spec, 'name', 'unknown'),
            estimated_total_cost=total_cost,
            cost_breakdown=cost_breakdown,
            cost_per_agent=cost_per_agent,
            cost_per_tool=cost_per_tool,
            estimated_tokens=estimated_tokens,
            estimated_duration_minutes=estimated_duration,
            confidence_score=confidence,
            created_at=datetime.now()
        )
    
    def analyze_historical_costs(self, crew_name: str, days: int = 30) -> Dict[str, Any]:
        """Analyze historical costs for a crew."""
        if not self.performance_tracker:
            return {"error": "Performance tracker not available"}
        
        # Get historical data
        performance_data = self.performance_tracker.get_crew_performance(crew_name, days)
        
        if performance_data['total_executions'] == 0:
            return {"error": "No historical data available"}
        
        # Calculate cost trends
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # This would require additional database queries to get historical cost data
        # For now, return basic analysis
        return {
            "crew_name": crew_name,
            "period_days": days,
            "total_executions": performance_data['total_executions'],
            "average_cost_per_execution": performance_data['average_cost'],
            "total_cost": performance_data['average_cost'] * performance_data['total_executions'],
            "cost_trend": "stable",  # Would calculate actual trend
            "cost_efficiency_score": self._calculate_efficiency_score(performance_data),
            "recommendations": self._generate_cost_recommendations(performance_data)
        }
    
    def compare_providers(self, crew_spec: Any) -> Dict[str, CostEstimate]:
        """Compare costs across different LLM providers."""
        estimates = {}
        
        for provider, models in self.PRICING_DATA.items():
            if not models:  # Skip empty providers
                continue
            
            # Use the first model for each provider
            model = list(models.keys())[0]
            estimates[provider] = self.estimate_crew_cost(crew_spec, provider, model)
        
        return estimates
    
    def optimize_for_cost(self, crew_spec: Any, target_cost: float) -> Dict[str, Any]:
        """Suggest optimizations to meet a target cost."""
        current_estimate = self.estimate_crew_cost(crew_spec)
        
        if current_estimate.estimated_total_cost <= target_cost:
            return {
                "meets_target": True,
                "current_cost": current_estimate.estimated_total_cost,
                "target_cost": target_cost,
                "savings": 0,
                "recommendations": []
            }
        
        recommendations = []
        potential_savings = 0
        
        # Suggest cheaper LLM models
        if current_estimate.cost_breakdown["llm_cost"] > target_cost * 0.5:
            cheaper_models = self._find_cheaper_models(crew_spec)
            if cheaper_models:
                recommendations.append({
                    "type": "llm_model",
                    "title": "Use cheaper LLM model",
                    "description": f"Switch to {cheaper_models[0]['model']} to save ${cheaper_models[0]['savings']:.4f}",
                    "savings": cheaper_models[0]['savings'],
                    "impact": "high"
                })
                potential_savings += cheaper_models[0]['savings']
        
        # Suggest reducing tool usage
        expensive_tools = self._find_expensive_tools(crew_spec)
        if expensive_tools:
            recommendations.append({
                "type": "tool_optimization",
                "title": "Optimize tool usage",
                "description": f"Reduce usage of expensive tools: {', '.join(expensive_tools)}",
                "savings": sum(tool['cost'] for tool in expensive_tools),
                "impact": "medium"
            })
            potential_savings += sum(tool['cost'] for tool in expensive_tools)
        
        # Suggest reducing agent iterations
        if hasattr(crew_spec, 'agents'):
            recommendations.append({
                "type": "agent_optimization",
                "title": "Reduce agent iterations",
                "description": "Limit max_iter to reduce LLM calls",
                "savings": current_estimate.cost_breakdown["llm_cost"] * 0.2,
                "impact": "medium"
            })
            potential_savings += current_estimate.cost_breakdown["llm_cost"] * 0.2
        
        return {
            "meets_target": potential_savings >= (current_estimate.estimated_total_cost - target_cost),
            "current_cost": current_estimate.estimated_total_cost,
            "target_cost": target_cost,
            "potential_savings": potential_savings,
            "recommendations": recommendations
        }
    
    def _estimate_tokens(self, crew_spec: Any) -> int:
        """Estimate total tokens needed for crew execution."""
        base_tokens = 1000  # Base prompt tokens
        
        # Add tokens for each agent
        agent_tokens = 0
        if hasattr(crew_spec, 'agents'):
            for agent in crew_spec.agents:
                # Estimate tokens for agent prompts
                agent_tokens += 500  # Role, goal, backstory
                agent_tokens += 200  # Tool descriptions
                agent_tokens += 300  # Task context
        
        # Add tokens for task execution
        task_tokens = 0
        if hasattr(crew_spec, 'task'):
            task_tokens += len(crew_spec.task.split()) * 2  # Rough estimate
        
        # Add tokens for iterations
        iteration_tokens = 0
        if hasattr(crew_spec, 'agents'):
            for agent in crew_spec.agents:
                max_iter = getattr(agent, 'max_iter', 5)
                iteration_tokens += max_iter * 200  # Tokens per iteration
        
        return base_tokens + agent_tokens + task_tokens + iteration_tokens
    
    def _calculate_tool_costs(self, crew_spec: Any) -> float:
        """Calculate total tool usage costs."""
        total_cost = 0.0
        
        if hasattr(crew_spec, 'agents'):
            for agent in crew_spec.agents:
                tools = getattr(agent, 'required_tools', [])
                for tool in tools:
                    tool_cost = self.TOOL_COSTS.get(tool, 0.001)  # Default cost
                    # Estimate 2-5 calls per tool per agent
                    estimated_calls = 3
                    total_cost += tool_cost * estimated_calls
        
        return total_cost
    
    def _calculate_agent_costs(self, crew_spec: Any, base_llm_cost: float) -> float:
        """Calculate additional costs for agent coordination."""
        # Additional 20% for agent coordination overhead
        return base_llm_cost * 0.2
    
    def _calculate_agent_tool_costs(self, agent: Any) -> float:
        """Calculate tool costs for a specific agent."""
        tools = getattr(agent, 'required_tools', [])
        total_cost = 0.0
        
        for tool in tools:
            tool_cost = self.TOOL_COSTS.get(tool, 0.001)
            estimated_calls = 3
            total_cost += tool_cost * estimated_calls
        
        return total_cost
    
    def _calculate_per_tool_costs(self, crew_spec: Any) -> Dict[str, float]:
        """Calculate costs per tool type."""
        tool_costs = {}
        
        if hasattr(crew_spec, 'agents'):
            for agent in crew_spec.agents:
                tools = getattr(agent, 'required_tools', [])
                for tool in tools:
                    if tool not in tool_costs:
                        tool_costs[tool] = 0.0
                    tool_costs[tool] += self.TOOL_COSTS.get(tool, 0.001) * 3
        
        return tool_costs
    
    def _estimate_duration(self, crew_spec: Any) -> float:
        """Estimate execution duration in minutes."""
        base_duration = 5.0  # Base 5 minutes
        
        # Add time for each agent
        if hasattr(crew_spec, 'agents'):
            for agent in crew_spec.agents:
                max_iter = getattr(agent, 'max_iter', 5)
                base_duration += max_iter * 2  # 2 minutes per iteration
        
        # Add time for tool usage
        tool_count = 0
        if hasattr(crew_spec, 'agents'):
            for agent in crew_spec.agents:
                tools = getattr(agent, 'required_tools', [])
                tool_count += len(tools)
        
        base_duration += tool_count * 0.5  # 30 seconds per tool
        
        return base_duration
    
    def _calculate_confidence_score(self, crew_spec: Any, provider: str, model: str) -> float:
        """Calculate confidence score for the cost estimate."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we have historical data
        if self.performance_tracker:
            # This would check for historical data
            confidence += 0.2
        
        # Adjust based on crew complexity
        if hasattr(crew_spec, 'agents'):
            agent_count = len(crew_spec.agents)
            if agent_count <= 2:
                confidence += 0.1
            elif agent_count >= 5:
                confidence -= 0.1
        
        # Adjust based on provider/model familiarity
        if provider in self.PRICING_DATA and model in self.PRICING_DATA[provider]:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_efficiency_score(self, performance_data: Dict[str, Any]) -> float:
        """Calculate cost efficiency score."""
        if performance_data['total_executions'] == 0:
            return 0.0
        
        # Simple efficiency score based on success rate and cost
        success_rate = performance_data['success_rate']
        avg_cost = performance_data['average_cost']
        
        # Higher success rate and lower cost = better efficiency
        efficiency = (success_rate * 0.7) + ((1.0 - min(avg_cost / 1.0, 1.0)) * 0.3)
        return min(1.0, max(0.0, efficiency))
    
    def _generate_cost_recommendations(self, performance_data: Dict[str, Any]) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        if performance_data['average_cost'] > 1.0:
            recommendations.append("Consider using a cheaper LLM model")
        
        if performance_data['success_rate'] < 0.8:
            recommendations.append("Improve crew reliability to reduce retry costs")
        
        if performance_data['average_duration_seconds'] > 300:  # 5 minutes
            recommendations.append("Optimize crew execution time to reduce compute costs")
        
        return recommendations
    
    def _find_cheaper_models(self, crew_spec: Any) -> List[Dict[str, Any]]:
        """Find cheaper LLM models for the crew."""
        # This would analyze the crew requirements and suggest cheaper alternatives
        # For now, return a simple example
        return [
            {
                "model": "gpt-3.5-turbo",
                "savings": 0.05,
                "provider": "openai"
            }
        ]
    
    def _find_expensive_tools(self, crew_spec: Any) -> List[Dict[str, Any]]:
        """Find expensive tools used by the crew."""
        expensive_tools = []
        
        if hasattr(crew_spec, 'agents'):
            for agent in crew_spec.agents:
                tools = getattr(agent, 'required_tools', [])
                for tool in tools:
                    cost = self.TOOL_COSTS.get(tool, 0.001)
                    if cost > 0.005:  # More than $0.005 per call
                        expensive_tools.append({
                            "tool": tool,
                            "cost": cost,
                            "agent": getattr(agent, 'name', 'unknown')
                        })
        
        return expensive_tools
