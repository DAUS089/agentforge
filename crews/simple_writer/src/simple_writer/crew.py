"""
simple_writer Crew Implementation

This module contains the main crew logic and orchestration.
"""

import yaml
import os
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from .tools.custom_tools import get_tools_for_agent


class SimpleWriterCrew:
    """Main crew class for simple_writer."""
    
    def __init__(self):
        """Initialize the crew."""
        self.config_path = Path(__file__).parent.parent.parent / "config"
        self.agents_config = self._load_config("agents.yaml")
        self.tasks_config = self._load_config("tasks.yaml")
        
        # Setup LLM configuration
        self.llm = self._setup_llm()
        
        # Initialize agents and tasks
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        
        # Create the crew
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            process=Process.sequential,
            verbose=True,
            memory=False  # Can be enabled as needed
        )
    
    def _load_config(self, filename: str) -> dict:
        """Load configuration from YAML file."""
        config_file = self.config_path / filename
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_llm(self) -> LLM:
        """Setup LLM configuration for CrewAI."""
        # Check for agentforge environment variables first
        provider = os.getenv('agentforge_LLM_PROVIDER', 'openai')
        model = os.getenv('agentforge_LLM_MODEL', 'gpt-4')
        api_key = os.getenv('agentforge_LLM_API_KEY')
        base_url = os.getenv('agentforge_LLM_BASE_URL')
        
        # If custom provider configuration exists, use it
        if provider == 'custom' and api_key and base_url:
            return LLM(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.7,
                max_tokens=2000
            )
        
        # Check if agents_config has LLM configuration
        if hasattr(self.agents_config, 'get') and self.agents_config.get('llm'):
            llm_config = self.agents_config['llm']
            return LLM(
                model=llm_config.get('model', 'gpt-4'),
                api_key=llm_config.get('api_key'),
                base_url=llm_config.get('base_url'),
                temperature=llm_config.get('temperature', 0.7),
                max_tokens=llm_config.get('max_tokens', 2000)
            )
        
        # Otherwise, use standard providers (OpenAI, Anthropic, etc.)
        # CrewAI will auto-detect based on environment variables
        if os.getenv('OPENAI_API_KEY'):
            return LLM(model='gpt-4', temperature=0.7)
        elif os.getenv('ANTHROPIC_API_KEY'):
            return LLM(model='claude-3-sonnet-20240229', temperature=0.7)
        elif os.getenv('GOOGLE_API_KEY'):
            return LLM(model='gemini-pro', temperature=0.7)
        else:
            # Default to OpenAI (will fail if no API key, but that's expected)
            return LLM(model='gpt-3.5-turbo', temperature=0.7)
    
    def _create_agent_llm(self, llm_config: dict) -> LLM:
        """Create LLM instance for a specific agent."""
        provider = llm_config.get('provider', 'openai')
        model = llm_config.get('model', 'gpt-4')
        
        # Extract all possible LLM parameters from config
        llm_params = {
            'model': model,
            'temperature': llm_config.get('temperature', 0.7),
            'max_tokens': llm_config.get('max_tokens'),
            'top_p': llm_config.get('top_p'),
            'frequency_penalty': llm_config.get('frequency_penalty'),
            'presence_penalty': llm_config.get('presence_penalty'),
            'stop': llm_config.get('stop'),
            'timeout': llm_config.get('timeout'),
            'max_retries': llm_config.get('max_retries'),
            'api_key': llm_config.get('api_key'),
            'base_url': llm_config.get('base_url'),
            'api_version': llm_config.get('api_version'),
            'organization': llm_config.get('organization')
        }
        
        # Remove None values to avoid passing them to LLM constructor
        llm_params = {k: v for k, v in llm_params.items() if v is not None}
        
        # Check if environment variables override the config
        env_provider = os.getenv('agentforge_LLM_PROVIDER')
        env_model = os.getenv('agentforge_LLM_MODEL')
        env_api_key = os.getenv('agentforge_LLM_API_KEY')
        env_base_url = os.getenv('agentforge_LLM_BASE_URL')
        
        # If environment variables are set, use them (highest priority)
        if env_provider and env_model:
            llm_params['model'] = env_model
            if env_provider == 'custom' and env_api_key and env_base_url:
                llm_params['api_key'] = env_api_key
                llm_params['base_url'] = env_base_url
            return LLM(**llm_params)
        
        # Otherwise use agent-specific config
        return LLM(**llm_params)
    
    def _create_agents(self) -> dict:
        """Create agents from configuration."""
        agents = {}
        
        # Check if agents_config is properly loaded
        if not self.agents_config:
            raise ValueError("agents_config is empty or not loaded properly")

        for agent_name, agent_config in self.agents_config.items():
            # Get tools for this agent
            tools = get_tools_for_agent(agent_config.get('tools', []))
            
            # Use agent-specific LLM config if available, otherwise use default
            agent_llm_config = agent_config.get('llm', {})
            if agent_llm_config:
                agent_llm = self._create_agent_llm(agent_llm_config)
            else:
                agent_llm = self.llm
            
            # Create agent with proper error handling
            try:
                agent = Agent(
                    role=agent_config.get('role', f'Agent {agent_name}'),
                    goal=agent_config.get('goal', 'Complete assigned tasks'),
                    backstory=agent_config.get('backstory', 'A helpful AI agent'),
                    llm=agent_llm,
                    tools=tools,
                    verbose=agent_config.get('verbose', True),
                    allow_delegation=agent_config.get('allow_delegation', False),
                    max_iter=agent_config.get('max_iter', 3),
                    max_execution_time=agent_config.get('max_execution_time')
                )
                agents[agent_name] = agent
            except Exception as e:
                print(f"Error creating agent {agent_name}: {e}")
                continue
        
        return agents
    
    def _create_tasks(self) -> dict:
        """Create tasks from configuration."""
        tasks = {}
        
        # Check if tasks_config is properly loaded
        if not self.tasks_config:
            raise ValueError("tasks_config is empty or not loaded properly")
        
        # First pass: Create all tasks without context
        for task_name, task_config in self.tasks_config.items():
            # Get the agent for this task
            agent_name = task_config['agent']
            agent = self.agents[agent_name]
            
            task = Task(
                description=task_config['description'],
                expected_output=task_config['expected_output'],
                agent=agent,
                context=None  # Will be set in second pass
            )
            
            tasks[task_name] = task
        
        # Second pass: Set up context relationships
        for task_name, task_config in self.tasks_config.items():
            context_tasks = []
            if 'context' in task_config:
                for context_task_name in task_config['context']:
                    if context_task_name in tasks:
                        context_tasks.append(tasks[context_task_name])
                        print(f"[INFO] Context linked: {task_name} <- {context_task_name}")
                    else:
                        print(f"[WARN] Context task '{context_task_name}' not found for task '{task_name}'")
            
            # Update task with context
            if context_tasks:
                tasks[task_name].context = context_tasks
                print(f"[INFO] Task '{task_name}' has {len(context_tasks)} context task(s)")
            else:
                print(f"[INFO] Task '{task_name}' has no context (root task)")
        
        return tasks
    
    def run(self, task_input: str = None) -> str:
        """Run the crew with optional task input."""
        try:
            # If task input is provided, update the main task description
            if task_input and task_input.strip():
                print(f"\n[INFO] Task Input Received: {task_input}")
                
                # Update the main task with the specific input
                if 'main_task' in self.tasks:
                    original_desc = self.tasks_config['main_task']['description']
                    enhanced_desc = f"{original_desc}\n\nSpecific Task: {task_input}"
                    self.tasks['main_task'].description = enhanced_desc
                    print(f"[INFO] Updated main task description with input")
                else:
                    print("[WARN] No main_task found to update")
            else:
                print("[INFO] No specific task input provided, using default task description")
            
            # Execute the crew
            print("\n[INFO] Starting crew execution...")
            result = self.crew.kickoff()
            return str(result)
            
        except Exception as e:
            error_msg = f"Crew execution failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg
    
    def get_crew_info(self) -> dict:
        """Get information about the crew configuration."""
        return {
            'name': 'simple_writer',
            'description': 'AI-orchestrated crew for: Create a simple text generator that writes basic content without external tools',
            'agents': list(self.agents.keys()),
            'tasks': list(self.tasks.keys()),
            'process_type': 'sequential'
        }
