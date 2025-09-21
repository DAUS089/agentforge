"""
Reinforcement Learning Agent Creator for AgentForge.

This module implements a sophisticated RL system that learns when and how to create
new agents based on performance feedback, task complexity, and environmental changes.
"""

import numpy as np
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path

from .config import Config
from .adaptive_agent_manager import AdaptiveAgentManager, AgentCreationTrigger
from ..logging.logger import get_logger

class RLState(Enum):
    """States in the RL environment."""
    NORMAL_OPERATION = "normal_operation"
    PERFORMANCE_DEGRADED = "performance_degraded"
    HIGH_COMPLEXITY_TASK = "high_complexity_task"
    NEW_CAPABILITY_NEEDED = "new_capability_needed"
    FAILURE_RECOVERY = "failure_recovery"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"

class RLAction(Enum):
    """Actions the RL agent can take."""
    NO_ACTION = "no_action"
    CREATE_SPECIALIST_AGENT = "create_specialist_agent"
    CREATE_OPTIMIZATION_AGENT = "create_optimization_agent"
    CREATE_CAPABILITY_AGENT = "create_capability_agent"
    CREATE_RECOVERY_AGENT = "create_recovery_agent"
    MODIFY_EXISTING_AGENT = "modify_existing_agent"
    WAIT_AND_OBSERVE = "wait_and_observe"

@dataclass
class RLExperience:
    """Experience tuple for RL learning."""
    state: RLState
    action: RLAction
    reward: float
    next_state: RLState
    done: bool
    metadata: Dict[str, Any]

class QLearningAgent:
    """Q-Learning agent for adaptive agent creation decisions."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 exploration_rate: float = 0.3, exploration_decay: float = 0.995):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.01
        
        # Q-table: state -> action -> value
        self.q_table = {}
        self.experience_buffer = []
        self.max_buffer_size = 10000
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.learning_history = []
        
        self.logger = get_logger("rl_agent_creator")
    
    def _get_state_key(self, state: RLState) -> str:
        """Convert state to string key for Q-table."""
        return state.value
    
    def _get_action_key(self, action: RLAction) -> str:
        """Convert action to string key for Q-table."""
        return action.value
    
    def _initialize_q_table(self):
        """Initialize Q-table with all state-action pairs."""
        for state in RLState:
            state_key = self._get_state_key(state)
            self.q_table[state_key] = {}
            for action in RLAction:
                action_key = self._get_action_key(action)
                self.q_table[state_key][action_key] = 0.0
    
    def get_q_value(self, state: RLState, action: RLAction) -> float:
        """Get Q-value for state-action pair."""
        if not self.q_table:
            self._initialize_q_table()
        
        state_key = self._get_state_key(state)
        action_key = self._get_action_key(action)
        
        return self.q_table.get(state_key, {}).get(action_key, 0.0)
    
    def update_q_value(self, state: RLState, action: RLAction, new_value: float):
        """Update Q-value for state-action pair."""
        if not self.q_table:
            self._initialize_q_table()
        
        state_key = self._get_state_key(state)
        action_key = self._get_action_key(action)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        self.q_table[state_key][action_key] = new_value
    
    def choose_action(self, state: RLState, available_actions: List[RLAction] = None) -> RLAction:
        """Choose action using epsilon-greedy policy."""
        if available_actions is None:
            available_actions = list(RLAction)
        
        # Exploration vs Exploitation
        if np.random.random() < self.exploration_rate:
            # Exploration: choose random action
            action = np.random.choice(available_actions)
            self.logger.debug("RL_ACTION", f"Exploration: chose {action.value} for state {state.value}")
        else:
            # Exploitation: choose best action
            action = self._get_best_action(state, available_actions)
            self.logger.debug("RL_ACTION", f"Exploitation: chose {action.value} for state {state.value}")
        
        return action
    
    def _get_best_action(self, state: RLState, available_actions: List[RLAction]) -> RLAction:
        """Get the best action for a given state."""
        best_action = available_actions[0]
        best_value = self.get_q_value(state, best_action)
        
        for action in available_actions[1:]:
            value = self.get_q_value(state, action)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def learn(self, experience: RLExperience):
        """Learn from an experience using Q-learning."""
        # Add experience to buffer
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        # Q-learning update
        current_q = self.get_q_value(experience.state, experience.action)
        
        if experience.done:
            # Terminal state
            target_q = experience.reward
        else:
            # Non-terminal state
            next_state_actions = [action for action in RLAction]  # All actions available
            max_next_q = max([self.get_q_value(experience.next_state, action) 
                            for action in next_state_actions])
            target_q = experience.reward + self.discount_factor * max_next_q
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.update_q_value(experience.state, experience.action, new_q)
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.exploration_rate * self.exploration_decay,
            self.min_exploration_rate
        )
        
        # Log learning progress
        self.learning_history.append({
            "episode": len(self.episode_rewards),
            "state": experience.state.value,
            "action": experience.action.value,
            "reward": experience.reward,
            "q_value": new_q,
            "exploration_rate": self.exploration_rate
        })
    
    def batch_learn(self, batch_size: int = 32):
        """Learn from a batch of experiences."""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample random batch
        batch = np.random.choice(self.experience_buffer, batch_size, replace=False)
        
        for experience in batch:
            self.learn(experience)
    
    def save_model(self, filepath: str):
        """Save the RL model to disk."""
        model_data = {
            "q_table": self.q_table,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "exploration_decay": self.exploration_decay,
            "learning_history": self.learning_history[-1000:],  # Keep last 1000 entries
            "episode_rewards": self.episode_rewards[-1000:],
            "episode_lengths": self.episode_lengths[-1000:]
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info("RL_MODEL", f"RL model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the RL model from disk."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data.get("q_table", {})
            self.learning_rate = model_data.get("learning_rate", 0.1)
            self.discount_factor = model_data.get("discount_factor", 0.95)
            self.exploration_rate = model_data.get("exploration_rate", 0.3)
            self.exploration_decay = model_data.get("exploration_decay", 0.995)
            self.learning_history = model_data.get("learning_history", [])
            self.episode_rewards = model_data.get("episode_rewards", [])
            self.episode_lengths = model_data.get("episode_lengths", [])
            
            self.logger.info("RL_MODEL", f"RL model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error("RL_MODEL", f"Failed to load RL model: {str(e)}")
            return False
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        if not self.episode_rewards:
            return {"status": "no_data"}
        
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 100 else self.episode_rewards
        recent_lengths = self.episode_lengths[-100:] if len(self.episode_lengths) > 100 else self.episode_lengths
        
        return {
            "status": "trained",
            "total_episodes": len(self.episode_rewards),
            "average_reward": np.mean(recent_rewards),
            "average_episode_length": np.mean(recent_lengths),
            "exploration_rate": self.exploration_rate,
            "q_table_size": sum(len(actions) for actions in self.q_table.values()),
            "learning_progress": self._calculate_learning_progress()
        }
    
    def _calculate_learning_progress(self) -> str:
        """Calculate learning progress based on recent performance."""
        if len(self.episode_rewards) < 10:
            return "early_stage"
        
        recent_rewards = self.episode_rewards[-10:]
        older_rewards = self.episode_rewards[-20:-10] if len(self.episode_rewards) >= 20 else self.episode_rewards[:10]
        
        recent_avg = np.mean(recent_rewards)
        older_avg = np.mean(older_rewards)
        
        improvement = (recent_avg - older_avg) / abs(older_avg) if older_avg != 0 else 0
        
        if improvement > 0.1:
            return "improving"
        elif improvement < -0.1:
            return "degrading"
        else:
            return "stable"

class RLAgentCreator:
    """Main RL-based agent creation system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("rl_agent_creator")
        self.adaptive_manager = AdaptiveAgentManager(config)
        
        # Initialize RL agent
        self.rl_agent = QLearningAgent()
        
        # State tracking
        self.current_state = RLState.NORMAL_OPERATION
        self.episode_start_time = time.time()
        self.episode_reward = 0.0
        self.episode_actions = []
        
        # Model persistence
        self.model_path = Path.home() / ".agentforge" / "rl_model.pkl"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing model if available
        self.rl_agent.load_model(str(self.model_path))
    
    def observe_environment(self, crew_name: str, task_context: Dict[str, Any]) -> RLState:
        """Observe the current environment and determine state."""
        try:
            # Analyze current performance
            performance_analysis = self.adaptive_manager.analyze_agent_performance(crew_name)
            
            if performance_analysis["status"] != "success":
                return RLState.NORMAL_OPERATION
            
            # Determine state based on performance and context
            overall_performance = performance_analysis["overall_performance"]["overall_score"]
            recommendations = performance_analysis["recommendations"]
            task_complexity = task_context.get("complexity", 0.5)
            
            # State determination logic
            if overall_performance < 0.6:
                if any(rec["issue"] == "low_success_rate" for rec in recommendations):
                    return RLState.FAILURE_RECOVERY
                else:
                    return RLState.PERFORMANCE_DEGRADED
            elif task_complexity > 0.8:
                return RLState.HIGH_COMPLEXITY_TASK
            elif any(rec["type"] == "quality_improvement" for rec in recommendations):
                return RLState.OPTIMIZATION_OPPORTUNITY
            elif task_context.get("required_capabilities"):
                current_capabilities = self.adaptive_manager._get_current_capabilities(performance_analysis)
                required_capabilities = task_context.get("required_capabilities", [])
                missing_capabilities = set(required_capabilities) - set(current_capabilities)
                if missing_capabilities:
                    return RLState.NEW_CAPABILITY_NEEDED
            else:
                return RLState.NORMAL_OPERATION
            
        except Exception as e:
            self.logger.error("RL_ENVIRONMENT", f"Error observing environment: {str(e)}")
            return RLState.NORMAL_OPERATION
    
    def get_available_actions(self, state: RLState) -> List[RLAction]:
        """Get available actions for a given state."""
        action_mapping = {
            RLState.NORMAL_OPERATION: [RLAction.NO_ACTION, RLAction.WAIT_AND_OBSERVE],
            RLState.PERFORMANCE_DEGRADED: [RLAction.CREATE_OPTIMIZATION_AGENT, RLAction.MODIFY_EXISTING_AGENT, RLAction.WAIT_AND_OBSERVE],
            RLState.HIGH_COMPLEXITY_TASK: [RLAction.CREATE_SPECIALIST_AGENT, RLAction.CREATE_CAPABILITY_AGENT, RLAction.WAIT_AND_OBSERVE],
            RLState.NEW_CAPABILITY_NEEDED: [RLAction.CREATE_CAPABILITY_AGENT, RLAction.MODIFY_EXISTING_AGENT, RLAction.WAIT_AND_OBSERVE],
            RLState.FAILURE_RECOVERY: [RLAction.CREATE_RECOVERY_AGENT, RLAction.MODIFY_EXISTING_AGENT, RLAction.WAIT_AND_OBSERVE],
            RLState.OPTIMIZATION_OPPORTUNITY: [RLAction.CREATE_OPTIMIZATION_AGENT, RLAction.MODIFY_EXISTING_AGENT, RLAction.WAIT_AND_OBSERVE]
        }
        
        return action_mapping.get(state, [RLAction.NO_ACTION])
    
    def execute_action(self, action: RLAction, crew_name: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the chosen action."""
        self.episode_actions.append(action)
        
        if action == RLAction.NO_ACTION:
            return {"status": "no_action", "reward": 0.0}
        
        elif action == RLAction.WAIT_AND_OBSERVE:
            return {"status": "wait", "reward": 0.1}  # Small positive reward for patience
        
        elif action in [RLAction.CREATE_SPECIALIST_AGENT, RLAction.CREATE_OPTIMIZATION_AGENT, 
                       RLAction.CREATE_CAPABILITY_AGENT, RLAction.CREATE_RECOVERY_AGENT]:
            # Create adaptive agent
            decision = self.adaptive_manager.decide_agent_creation(crew_name, task_context)
            
            if decision:
                result = self.adaptive_manager.create_adaptive_agent(decision)
                
                if result["status"] == "success":
                    return {
                        "status": "agent_created",
                        "agent_name": result["agent_name"],
                        "specialization": result["specialization"],
                        "reward": result["expected_improvement"] * 10  # Scale reward
                    }
                else:
                    return {"status": "creation_failed", "reward": -1.0}
            else:
                return {"status": "no_decision", "reward": -0.5}
        
        elif action == RLAction.MODIFY_EXISTING_AGENT:
            # Modify existing agent (placeholder)
            return {"status": "modification_planned", "reward": 0.5}
        
        else:
            return {"status": "unknown_action", "reward": -0.1}
    
    def calculate_reward(self, action: RLAction, result: Dict[str, Any], 
                        next_state: RLState, crew_name: str) -> float:
        """Calculate reward for the action taken."""
        base_reward = result.get("reward", 0.0)
        
        # Additional reward factors
        if action == RLAction.NO_ACTION and next_state == RLState.NORMAL_OPERATION:
            base_reward += 0.2  # Reward for maintaining good state
        
        elif action in [RLAction.CREATE_SPECIALIST_AGENT, RLAction.CREATE_OPTIMIZATION_AGENT,
                       RLAction.CREATE_CAPABILITY_AGENT, RLAction.CREATE_RECOVERY_AGENT]:
            if result["status"] == "agent_created":
                # Reward based on expected improvement
                base_reward += result.get("reward", 0.0)
            else:
                base_reward -= 0.5  # Penalty for failed creation
        
        # State transition rewards
        if next_state == RLState.NORMAL_OPERATION:
            base_reward += 0.3  # Reward for reaching normal state
        elif next_state == RLState.FAILURE_RECOVERY:
            base_reward -= 0.2  # Penalty for reaching failure state
        
        return base_reward
    
    def step(self, crew_name: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one RL step."""
        try:
            # Observe current state
            current_state = self.observe_environment(crew_name, task_context)
            
            # Get available actions
            available_actions = self.get_available_actions(current_state)
            
            # Choose action
            action = self.rl_agent.choose_action(current_state, available_actions)
            
            # Execute action
            result = self.execute_action(action, crew_name, task_context)
            
            # Observe next state
            next_state = self.observe_environment(crew_name, task_context)
            
            # Calculate reward
            reward = self.calculate_reward(action, result, next_state, crew_name)
            self.episode_reward += reward
            
            # Create experience
            experience = RLExperience(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=False,  # Episodes are continuous
                metadata={
                    "crew_name": crew_name,
                    "task_context": task_context,
                    "result": result
                }
            )
            
            # Learn from experience
            self.rl_agent.learn(experience)
            
            # Update state
            self.current_state = next_state
            
            return {
                "status": "success",
                "state": current_state.value,
                "action": action.value,
                "next_state": next_state.value,
                "reward": reward,
                "result": result,
                "episode_reward": self.episode_reward
            }
            
        except Exception as e:
            self.logger.error("RL_STEP", f"Error in RL step: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def end_episode(self):
        """End current episode and save learning progress."""
        # Record episode statistics
        self.rl_agent.episode_rewards.append(self.episode_reward)
        self.rl_agent.episode_lengths.append(len(self.episode_actions))
        
        # Batch learning
        self.rl_agent.batch_learn()
        
        # Save model
        self.rl_agent.save_model(str(self.model_path))
        
        # Reset episode
        self.episode_reward = 0.0
        self.episode_actions = []
        self.episode_start_time = time.time()
        
        self.logger.info("RL_EPISODE", f"Episode ended. Total reward: {self.episode_reward:.2f}")
    
    def get_rl_insights(self) -> Dict[str, Any]:
        """Get RL system insights."""
        rl_stats = self.rl_agent.get_learning_stats()
        
        return {
            "rl_agent_stats": rl_stats,
            "current_state": self.current_state.value,
            "episode_reward": self.episode_reward,
            "episode_actions": len(self.episode_actions),
            "model_path": str(self.model_path),
            "exploration_rate": self.rl_agent.exploration_rate
        }
