"""
AgentForge CLI - Forge intelligent AI agents with style and creativity.
"""

import warnings
import os
import time
import json
# Suppress common deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*langchain.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Pydantic.*deprecated.*") 
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
warnings.filterwarnings("ignore", message=".*event loop.*")
warnings.filterwarnings("ignore", message=".*extra keyword arguments.*")
warnings.filterwarnings("ignore", message=".*Field.*deprecated.*")
# Set environment variable to suppress additional warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Apply event bus patch to fix CrewAI EventBus errors
try:
    from .core.event_bus_patch import apply_patch
    apply_patch()
except ImportError:
    pass

import typer
import json
from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

from .core.config import Config
from .core.llm_provider import LLMProviderFactory
from .core.file_generator import CrewFileGenerator
from .core.master_agent_crew import MasterAgentCrew
from .core.task_analyzer import CrewSpec, AgentSpec
from .core.crew_designer import CrewModel
from .templates.template_manager import TemplateManager
from .analytics.performance_tracker import PerformanceTracker
from .analytics.cost_analyzer import CostAnalyzer
from .analytics.optimization_engine import OptimizationEngine
from .logging.logger import get_logger, setup_logging, LogLevel
from .logging.error_handler import get_error_handler, ErrorSeverity
from .logging.debug_tracer import get_tracer
from .core.adaptive_agent_manager import AdaptiveAgentManager
from .core.rl_agent_creator import RLAgentCreator, QLearningAgent, RLState
import numpy as np

app = typer.Typer(
    name="agentforge",
    help="""[bold cyan]🔥 AgentForge: Forge intelligent AI agents with CrewAI[/bold cyan]

[green]⚡ Quick Start:[/green]
  [cyan]agentforge forge[/cyan] "Create a blog writer who can write simple and informative blog posts for beginners." --name blog_writer_01 # FORGE
  [cyan]agentforge ignite[/cyan] blog_writer_01 --input "Write a blog post about the benefits of AI" # IGNITE
""",
    rich_markup_mode="rich"
)

console = Console()

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """Main callback that shows banner when no command is provided."""
    if ctx.invoked_subcommand is None:
        display_banner()

def display_banner():
    """Display AgentForge banner."""
    banner = """[bold cyan]
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║  [bold red]    █████╗  ███████╗███████╗███╗   ██╗████████╗[/bold red]  [bold blue]███████╗ ██████╗ ██████╗  ██████╗ ███████╗[/bold blue]  ║
    ║  [bold red]   ██╔══██╗██╔════╝██╔════╝████╗  ██║╚══██╔══╝[/bold red]  [bold blue]██╔════╝██╔═══██╗██╔══██╗██╔═══██╗██╔════╝[/bold blue]  ║
    ║  [bold red]   ███████║███████╗█████╗  ██╔██╗ ██║   ██║   [/bold red]  [bold blue]█████╗  ██║   ██║██████╔╝██║   ██║█████╗  [/bold blue]  ║
    ║  [bold red]   ██╔══██║╚════██║██╔══╝  ██║╚██╗██║   ██║   [/bold red]  [bold blue]██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  [/bold blue]  ║
    ║  [bold red]   ██║  ██║███████║███████╗██║ ╚████║   ██║   [/bold red]  [bold blue]██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗[/bold blue]  ║
    ║  [bold red]   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   [/bold red]  [bold blue]╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝[/bold blue]  ║
    ║                                                                              ║
    ║  [bold yellow]🔥 Forge intelligent AI agents with CrewAI[/bold yellow]                    ║
    ║  [dim]⚡ Transform ideas into powerful multi-agent systems[/dim]                        ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
"""

    console.print(banner)
    
    console.print(f"\n[bold yellow]⚡ Getting Started[/bold yellow]")
    console.print("🔥" + "=" * 58 + "🔥")
    
    console.print("\n[bold green]🔥 Step 1:[/bold green] Forge your first agent crew")
    console.print("  [cyan]agentforge forge \"A blog writer who can write simple and informative blog posts for beginners.\" --name blog_writer_01[/cyan]")
    console.print("  [dim]⚒️  Forges: YAML configs, Python modules, documentation[/dim]")
    
    console.print("\n[bold green]🔥 Step 2:[/bold green] Ignite your crew (requires API key)")
    console.print("  [cyan]export OPENAI_API_KEY=\"your-key\"[/cyan]  # OpenAI")
    console.print("  [cyan]export ANTHROPIC_API_KEY=\"your-key\"[/cyan]  # Anthropic")
    
    console.print("\n[bold green]🔥 Step 3:[/bold green] Work with forged files")
    console.print("  [cyan]agentforge ignite blog_writer_01 --input \"Write a blog post about the benefits of AI\"[/cyan]")
    console.print("  [cyan]cd crews/blog_writer_01 && ./run.sh 'your input'[/cyan] # Alternative execution")
    console.print("  [dim]🔄 Version control friendly - track changes in Git[/dim]")
    
    console.print("\n[dim]💡 Essential Commands:[/dim]")
    console.print("[dim]  • agentforge forge \"task\" - Forge new agent crew[/dim]")
    console.print("[dim]  • agentforge ignite <name> - Ignite crew execution[/dim]")
    console.print("[dim]  • agentforge providers - Configure LLM providers[/dim]")
    console.print("[dim]  • agentforge adaptive analyze --crew <name> - Analyze adaptive opportunities[/dim]")
    console.print("[dim]  • agentforge rl train --crew <name> --episodes 50 - Train RL agent[/dim]")
    console.print("[dim]  • agentforge version - Show version[/dim]")

@app.command()
def forge(
    task: str = typer.Argument(..., help="Description of the task to accomplish"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Optional name for the crew"),
    template: Optional[str] = typer.Option(None, "--template", "-t", help="Use a specific template as starting point"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """🔥 Forge a new agent crew for a given task.
    
    Creates a self-contained CrewAI project with YAML configurations and Python modules.
    Can use templates as starting points for common use cases.
    
    Examples:
      agentforge forge "research competitors and write analysis report"
      agentforge forge "analyze cryptocurrency market trends" --name crypto_crew
      agentforge forge "analyze sales data" --template data_analysis
    """
    if verbose:
        display_banner()
    
    console.print(f"\n[bold green]🔥 Forging agent crew for task:[/bold green] {task}")
    
    try:
        config = Config()
        template_manager = TemplateManager()
        
        # Check if template is specified
        if template:
            template_obj = template_manager.get_template(template)
            if not template_obj:
                console.print(f"[red]❌ Template '{template}' not found[/red]")
                console.print(f"[yellow]Available templates: {', '.join(template_manager.list_templates()[:5])}...[/yellow]")
                raise typer.Exit(1)
            
            console.print(f"[dim]📋 Using template: {template_obj.name}[/dim]")
            console.print(f"[dim]Category: {template_obj.category} | Complexity: {template_obj.complexity}[/dim]")
        
        # Use master agents to analyze the task and generate crew specification
        console.print("[dim]🤖 Using AI master agents to analyze task and design crew...[/dim]")
        
        master_crew = MasterAgentCrew(config)
        crew_model = master_crew.create_crew(task, crew_name=name, verbose=verbose, use_ai_orchestration=True)
        
        # Convert CrewModel to CrewSpec for file generation compatibility
        from .core.task_analyzer import TaskComplexity
        
        if template:
            # Use template as base and customize for the specific task
            template_obj = template_manager.get_template(template)
            crew_spec = CrewSpec(
                name=name or f"{template_obj.name} - {task[:30]}...",
                task=task,
                description=f"{template_obj.description} - Customized for: {task}",
                agents=[
                    AgentSpec(
                        role=agent.role,
                        name=agent.name,
                        goal=f"{agent.goal} - Focus on: {task}",
                        backstory=agent.backstory,
                        required_tools=agent.tools,
                        memory_type=agent.memory_type,
                        max_iter=agent.max_iter,
                        allow_delegation=False
                    )
                    for agent in template_obj.agents
                ],
                expected_output=f"Complete results for: {task}",
                complexity=TaskComplexity.MODERATE if template_obj.complexity == "medium" else TaskComplexity.HIGH if template_obj.complexity == "high" else TaskComplexity.LOW,
                estimated_time=15,
                process_type=template_obj.workflow.value
            )
        else:
            # Use AI-generated crew specification
            crew_spec = CrewSpec(
                name=crew_model.name,
                task=crew_model.task,
                description=crew_model.description,
                agents=[
                    AgentSpec(
                        role=agent.role,
                        name=agent.name,
                        goal=agent.goal,
                        backstory=agent.backstory,
                        required_tools=getattr(agent, 'required_tools', []) or ['WebsiteSearchTool', 'FileReadTool'],
                        memory_type=getattr(agent, 'memory_type', 'short_term'),
                        max_iter=getattr(agent, 'max_iter', 5),
                        allow_delegation=getattr(agent, 'allow_delegation', False)
                    )
                    for agent in crew_model.agents
                ],
                expected_output=getattr(crew_model, 'expected_output', f"Complete results for: {crew_model.task}"),
                complexity=TaskComplexity.MODERATE,  # Default since master agents determine this
                estimated_time=15,  # Default
                process_type="sequential"
            )
        
        if not crew_spec:
            console.print("[red]❌ Failed to create crew using master agents[/red]")
            raise typer.Exit(1)
        
        # Generate file-based crew
        console.print("[dim]⚒️  Forging file-based crew structure...[/dim]")
        file_generator = CrewFileGenerator()
        crew_path = file_generator.generate_crew_project(crew_spec)
        
        console.print(f"\n[bold green]🔥 Forged Agent Crew:[/bold green] {crew_spec.name}")
        console.print(f"[bold blue]⚒️  Crew Path:[/bold blue] {crew_path}")
        
        # Display agents summary
        console.print(f"\n[bold blue]🤖 Forged Agents ({len(crew_spec.agents)}):[/bold blue]")
        for i, agent in enumerate(crew_spec.agents, 1):
            console.print(f"  {i}. [green]{agent.name}[/green] - {agent.role}")
        
        console.print(f"\n[bold green]⚒️  Forged Files:[/bold green]")
        console.print(f"  [cyan]•[/cyan] config/agents.yaml - Agent configurations")
        console.print(f"  [cyan]•[/cyan] config/tasks.yaml - Task definitions")
        console.print(f"  [cyan]•[/cyan] src/{crew_spec.name}/crew.py - Main crew logic")
        console.print(f"  [cyan]•[/cyan] src/{crew_spec.name}/main.py - Entry point")
        console.print(f"  [cyan]•[/cyan] requirements.txt - Dependencies")
        console.print(f"  [cyan]•[/cyan] run.sh - Execution script")
        console.print(f"  [cyan]•[/cyan] README.md - Documentation")
        
        console.print(f"\n[dim]💡 Next steps:[/dim]")
        console.print(f"[dim]  • agentforge ignite {crew_spec.name} --input \"your input\"[/dim]")
        console.print(f"[dim]  • cd {crew_path} && ./run.sh \"your input\"[/dim]")

    except Exception as e:
        console.print(f"\n[bold red]💥 Error forging crew:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def ignite(
    crew_name: str = typer.Argument(..., help="Name of the crew to ignite"),
    input_data: Optional[str] = typer.Option(None, "--input", "-i", help="Additional input data for the task"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """🔥 Ignite an existing crew to perform the task.
    
    Runs the crew in its generated project directory.
    
    Requirements:
    - OpenAI/Anthropic API key: export OPENAI_API_KEY="your-key"
    - Existing crew (forged with 'agentforge forge')
    
    Example: agentforge ignite my_research_crew --input "focus on recent data"
    """
    console.print(f"\n[bold green]🔥 Igniting crew:[/bold green] {crew_name}")
    if input_data:
        console.print(f"[bold blue]⚡ With additional context:[/bold blue] {input_data}")
    
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Find crew directory
        crews_base_path = Path("crews")
        crew_path = crews_base_path / crew_name
        
        if not crew_path.exists():
            console.print(f"[red]❌ Crew '{crew_name}' not found at {crew_path}[/red]")
            console.print("[dim]Available crews:[/dim]")
            if crews_base_path.exists():
                for crew_dir in crews_base_path.iterdir():
                    if crew_dir.is_dir():
                        console.print(f"[dim]  • {crew_dir.name}[/dim]")
            else:
                console.print("[dim]  No crews directory found[/dim]")
            raise typer.Exit(1)
        
        # Check if main.py exists
        main_py_path = crew_path / "src" / crew_name / "main.py"
        if not main_py_path.exists():
            console.print(f"[red]❌ Crew main.py not found at {main_py_path}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[dim]📁 Executing crew from: {crew_path}[/dim]")
        
        # Build command
        cmd = [sys.executable, "-m", f"{crew_name}.main"]
        if input_data:
            cmd.append(input_data)
        
        # Execute the crew with real-time output
        console.print(f"\n[bold green]🔥 Crew ignition successful![/bold green]")
        console.print(Panel("Starting crew execution...", title="⚡ Result", border_style="green"))
        
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            cwd=crew_path / "src",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())  # Print immediately
                output_lines.append(line)
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            console.print(f"\n[bold green]✅ Crew execution completed successfully![/bold green]")
        else:
            console.print(f"\n[bold red]💥 Crew execution failed with return code {process.returncode}[/bold red]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"\n[bold red]💥 Error igniting crew:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def providers(
    configure: Optional[str] = typer.Option(None, "--configure", "-c", help="Configure provider (openai, anthropic, google, deepseek, ollama, llamacpp, custom)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for provider (not needed for ollama/llamacpp)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Base URL for provider endpoint (optional for standard providers)"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name for provider"),
    model_path: Optional[str] = typer.Option(None, "--model-path", help="Path to model file (for llamacpp)"),
    ollama_host: Optional[str] = typer.Option(None, "--ollama-host", help="Ollama server URL (default: http://localhost:11434)")
):
    """Show available LLM providers and configuration examples.
    
    Lists all supported providers (OpenAI, Anthropic, Google, DeepSeek, Ollama, LlamaCpp, Custom)
    with their required environment variables and example configurations.
    """
    
    # Handle configuration if requested
    if configure:
        provider_name = configure.lower()
        supported_providers = ["openai", "anthropic", "google", "deepseek", "ollama", "llamacpp", "custom"]
        
        if provider_name not in supported_providers:
            console.print(f"[red]❌ Unsupported provider: {configure}[/red]")
            console.print(f"[yellow]Supported providers: {', '.join(supported_providers)}[/yellow]")
            raise typer.Exit(1)
        
        # Validate required parameters based on provider
        if provider_name in ["openai", "anthropic", "google", "deepseek", "custom"]:
            if not api_key or not model:
                console.print("[red]❌ For this provider, both --api-key and --model are required[/red]")
                raise typer.Exit(1)
        elif provider_name == "ollama":
            if not model:
                console.print("[red]❌ For Ollama, --model is required[/red]")
                raise typer.Exit(1)
        elif provider_name == "llamacpp":
            if not model or not model_path:
                console.print("[red]❌ For LlamaCpp, both --model and --model-path are required[/red]")
                raise typer.Exit(1)
        
        config = Config()
        
        # Set default base URLs for standard providers if not provided
        if base_url is None:
            default_base_urls = {
                "openai": "https://api.openai.com/v1",
                "anthropic": "https://api.anthropic.com/v1", 
                "google": "https://generativelanguage.googleapis.com/v1beta",
                "deepseek": "https://api.deepseek.com/v1",
                "ollama": ollama_host or "http://localhost:11434"
            }
            if provider_name == "custom":
                console.print("[red]❌ Custom provider requires --base-url parameter[/red]")
                raise typer.Exit(1)
            elif provider_name == "llamacpp":
                # LlamaCpp doesn't use base_url
                pass
            else:
                base_url = default_base_urls.get(provider_name)
        
        # Configure the provider
        config._config.llm.provider = provider_name
        config._config.llm.model = model
        
        if provider_name in ["openai", "anthropic", "google", "deepseek", "custom"]:
            config._config.llm.api_key = api_key
            config._config.llm.base_url = base_url
        elif provider_name == "ollama":
            config._config.llm.ollama_host = ollama_host or "http://localhost:11434"
        elif provider_name == "llamacpp":
            config._config.llm.model_path = model_path
        
        config.save_config()
        
        console.print(f"[green]✅ {provider_name.title()} provider configured successfully![/green]")
        console.print(f"[dim]Provider: {provider_name}[/dim]")
        if base_url:
            console.print(f"[dim]Base URL: {base_url}[/dim]")
        if model_path:
            console.print(f"[dim]Model Path: {model_path}[/dim]")
        if ollama_host:
            console.print(f"[dim]Ollama Host: {ollama_host}[/dim]")
        console.print(f"[dim]Model: {model}[/dim]")
        console.print(f"[dim]Config saved to: {config.config_path}[/dim]")
        return
    
    console.print("\n[bold blue]🔧 Available LLM Providers[/bold blue]")
    
    try:        
        console.print(f"\n[bold green]🔧 CLI Configuration (All Providers):[/bold green]")
        console.print("[bold]OpenAI:[/bold]")
        console.print("• [cyan]agentforge providers --configure openai --api-key \"your-openai-key\" --model \"gpt-4\"[/cyan]")
        console.print()
        console.print("[bold]Anthropic:[/bold]")
        console.print("• [cyan]agentforge providers --configure anthropic --api-key \"your-anthropic-key\" --model \"claude-3-sonnet-20240229\"[/cyan]")
        console.print()
        console.print("[bold]Google:[/bold]")
        console.print("• [cyan]agentforge providers --configure google --api-key \"your-google-key\" --model \"gemini-pro\"[/cyan]")
        console.print()
        console.print("[bold]DeepSeek:[/bold]")
        console.print("• [cyan]agentforge providers --configure deepseek --api-key \"your-deepseek-key\" --model \"deepseek-chat\"[/cyan]")
        console.print()
        console.print("[bold]Ollama (Local):[/bold]")
        console.print("• [cyan]agentforge providers --configure ollama --model \"llama3.1\" --ollama-host \"http://localhost:11434\"[/cyan]")
        console.print()
        console.print("[bold]LlamaCpp (Local):[/bold]")
        console.print("• [cyan]agentforge providers --configure llamacpp --model \"llama-3.1-8b\" --model-path \"/path/to/model.gguf\"[/cyan]")
        console.print()
        console.print("[bold]Custom Provider:[/bold]")
        console.print("• [cyan]agentforge providers --configure custom --api-key \"your-key\" --base-url \"https://api.example.com/v1\" --model \"gpt-4o-mini\"[/cyan]")
        
        config = Config()
        console.print(f"\n[dim]💡 Current provider: {config.llm.provider}[/dim]")
        console.print(f"[dim]💡 Current model: {config.llm.model}[/dim]")
        console.print(f"[dim]💡 Edit advanced settings in: {config.config_path}[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Error showing providers: {str(e)}[/red]")

@app.command()
def templates(
    list_templates: bool = typer.Option(False, "--list", "-l", help="List all available templates"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search templates by query"),
    show: Optional[str] = typer.Option(None, "--show", help="Show details of a specific template"),
    recommend: Optional[str] = typer.Option(None, "--recommend", "-r", help="Get template recommendations for a task"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter templates by category")
):
    """Manage crew templates and patterns.
    
    List, search, and get recommendations for crew templates.
    """
    template_manager = TemplateManager()
    
    try:
        if list_templates:
            templates = template_manager.list_templates()
            console.print(f"\n[bold blue]📋 Available Templates ({len(templates)})[/bold blue]")
            
            for template_name in templates:
                template = template_manager.get_template(template_name)
                if template:
                    console.print(f"\n[bold cyan]{template.name}[/bold cyan]")
                    console.print(f"[dim]Category: {template.category} | Complexity: {template.complexity} | Duration: {template.estimated_duration}[/dim]")
                    console.print(f"[dim]{template.description}[/dim]")
                    console.print(f"[dim]Use cases: {', '.join(template.use_cases[:3])}[/dim]")
        
        elif search:
            results = template_manager.search_templates(search)
            console.print(f"\n[bold blue]🔍 Search Results for '{search}' ({len(results)})[/bold blue]")
            
            for template in results:
                console.print(f"\n[bold cyan]{template.name}[/bold cyan]")
                console.print(f"[dim]Category: {template.category} | Complexity: {template.complexity}[/dim]")
                console.print(f"[dim]{template.description}[/dim]")
        
        elif show:
            template = template_manager.get_template(show)
            if template:
                console.print(f"\n[bold blue]📋 Template: {template.name}[/bold blue]")
                console.print(f"[dim]Category: {template.category}[/dim]")
                console.print(f"[dim]Workflow: {template.workflow.value}[/dim]")
                console.print(f"[dim]Complexity: {template.complexity}[/dim]")
                console.print(f"[dim]Duration: {template.estimated_duration}[/dim]")
                console.print(f"\n[bold]Description:[/bold] {template.description}")
                
                console.print(f"\n[bold]Agents ({len(template.agents)}):[/bold]")
                for agent in template.agents:
                    console.print(f"  • [cyan]{agent.name}[/cyan] - {agent.role}")
                    console.print(f"    [dim]Tools: {', '.join(agent.tools[:3])}{'...' if len(agent.tools) > 3 else ''}[/dim]")
                
                console.print(f"\n[bold]Tasks ({len(template.tasks)}):[/bold]")
                for task in template.tasks:
                    console.print(f"  • [cyan]{task.name}[/cyan] - {task.description}")
                
                console.print(f"\n[bold]Use Cases:[/bold] {', '.join(template.use_cases)}")
            else:
                console.print(f"[red]❌ Template '{show}' not found[/red]")
        
        elif recommend:
            recommendations = template_manager.get_template_recommendations(recommend)
            console.print(f"\n[bold blue]💡 Template Recommendations for: '{recommend}'[/bold blue]")
            
            if recommendations:
                for i, template in enumerate(recommendations, 1):
                    console.print(f"\n[bold cyan]{i}. {template.name}[/bold cyan]")
                    console.print(f"[dim]Category: {template.category} | Complexity: {template.complexity}[/dim]")
                    console.print(f"[dim]{template.description}[/dim]")
                    console.print(f"[dim]Use cases: {', '.join(template.use_cases[:2])}[/dim]")
            else:
                console.print("[yellow]No specific recommendations found. Try a more specific task description.[/yellow]")
        
        elif category:
            templates = template_manager.get_templates_by_category(category)
            console.print(f"\n[bold blue]📋 Templates in '{category}' category ({len(templates)})[/bold blue]")
            
            for template in templates:
                console.print(f"\n[bold cyan]{template.name}[/bold cyan]")
                console.print(f"[dim]Complexity: {template.complexity} | Duration: {template.estimated_duration}[/dim]")
                console.print(f"[dim]{template.description}[/dim]")
        
        else:
            # Show help
            console.print("\n[bold blue]📋 Crew Templates[/bold blue]")
            console.print("\n[bold]Available commands:[/bold]")
            console.print("  [cyan]--list[/cyan] - List all available templates")
            console.print("  [cyan]--search <query>[/cyan] - Search templates by query")
            console.print("  [cyan]--show <template>[/cyan] - Show details of a specific template")
            console.print("  [cyan]--recommend <task>[/cyan] - Get template recommendations for a task")
            console.print("  [cyan]--category <category>[/cyan] - Filter templates by category")
            
            # Show statistics
            stats = template_manager.get_template_statistics()
            console.print(f"\n[dim]Total templates: {stats['total_templates']} (built-in: {stats['built_in_templates']}, custom: {stats['custom_templates']})[/dim]")
            console.print(f"[dim]Categories: {', '.join(stats['categories'].keys())}[/dim]")
    
    except Exception as e:
        console.print(f"[red]❌ Error managing templates: {str(e)}[/red]")

@app.command()
def analytics(
    crew_name: Optional[str] = typer.Option(None, "--crew", "-c", help="Analyze specific crew"),
    days: int = typer.Option(30, "--days", "-d", help="Number of days to analyze"),
    show_summary: bool = typer.Option(False, "--summary", "-s", help="Show performance summary"),
    show_costs: bool = typer.Option(False, "--costs", help="Show cost analysis"),
    show_optimizations: bool = typer.Option(False, "--optimize", "-o", help="Show optimization recommendations"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Export data to file (JSON)")
):
    """Analyze crew performance and generate insights.
    
    Provides comprehensive analytics including performance metrics, cost analysis,
    and optimization recommendations.
    """
    try:
        performance_tracker = PerformanceTracker()
        cost_analyzer = CostAnalyzer(performance_tracker)
        optimization_engine = OptimizationEngine(performance_tracker, cost_analyzer)
        
        if show_summary:
            # Show overall performance summary
            summary = performance_tracker.get_performance_summary(days)
            console.print(f"\n[bold blue]📊 Performance Summary (Last {days} days)[/bold blue]")
            console.print(f"Total Executions: {summary['total_executions']}")
            console.print(f"Success Rate: {summary['overall_success_rate']:.1%}")
            console.print(f"Average Duration: {summary['average_duration_seconds']:.1f}s")
            console.print(f"Total Cost: ${summary['total_cost']:.2f}")
            console.print(f"Unique Crews: {summary['unique_crews']}")
            
            if summary['top_performing_crews']:
                console.print(f"\n[bold]Top Performing Crews:[/bold]")
                for i, crew in enumerate(summary['top_performing_crews'][:3], 1):
                    console.print(f"  {i}. [cyan]{crew['crew_name']}[/cyan] - {crew['success_rate']:.1%} success rate")
        
        elif crew_name:
            # Analyze specific crew
            console.print(f"\n[bold blue]📊 Analyzing Crew: {crew_name}[/bold blue]")
            
            # Performance analysis
            performance = performance_tracker.get_crew_performance(crew_name, days)
            console.print(f"\n[bold]Performance Metrics:[/bold]")
            console.print(f"Total Executions: {performance['total_executions']}")
            console.print(f"Success Rate: {performance['success_rate']:.1%}")
            console.print(f"Average Duration: {performance['average_duration_seconds']:.1f}s")
            console.print(f"Average Cost: ${performance['average_cost']:.2f}")
            console.print(f"Average Quality: {performance['average_quality_score']:.1f}/10")
            
            if show_costs:
                # Cost analysis
                cost_analysis = cost_analyzer.analyze_historical_costs(crew_name, days)
                if "error" not in cost_analysis:
                    console.print(f"\n[bold]Cost Analysis:[/bold]")
                    console.print(f"Total Cost: ${cost_analysis['total_cost']:.2f}")
                    console.print(f"Cost per Execution: ${cost_analysis['average_cost_per_execution']:.2f}")
                    console.print(f"Efficiency Score: {cost_analysis['cost_efficiency_score']:.1f}/10")
                    
                    if cost_analysis['recommendations']:
                        console.print(f"\n[bold]Cost Recommendations:[/bold]")
                        for rec in cost_analysis['recommendations']:
                            console.print(f"  • {rec}")
            
            if show_optimizations:
                # Optimization recommendations
                recommendations = optimization_engine.analyze_crew_performance(crew_name, days)
                if recommendations:
                    console.print(f"\n[bold]Optimization Recommendations:[/bold]")
                    for i, rec in enumerate(recommendations[:5], 1):
                        console.print(f"\n{i}. [cyan]{rec.title}[/cyan]")
                        console.print(f"   [dim]Impact: {rec.impact_score:.1f}/10 | Effort: {rec.implementation_effort}[/dim]")
                        console.print(f"   {rec.description}")
                        if rec.implementation_steps:
                            console.print(f"   [dim]Steps: {rec.implementation_steps[0]}[/dim]")
                else:
                    console.print("[yellow]No optimization recommendations available[/yellow]")
        
        else:
            # Show help
            console.print("\n[bold blue]📊 Crew Analytics[/bold blue]")
            console.print("\n[bold]Available commands:[/bold]")
            console.print("  [cyan]--summary[/cyan] - Show overall performance summary")
            console.print("  [cyan]--crew <name>[/cyan] - Analyze specific crew")
            console.print("  [cyan]--costs[/cyan] - Include cost analysis")
            console.print("  [cyan]--optimize[/cyan] - Include optimization recommendations")
            console.print("  [cyan]--days <number>[/cyan] - Number of days to analyze")
            console.print("  [cyan]--export <file>[/cyan] - Export data to JSON file")
        
        if export:
            # Export data
            data = {
                "summary": performance_tracker.get_performance_summary(days),
                "crews": {}
            }
            
            if crew_name:
                data["crews"][crew_name] = {
                    "performance": performance_tracker.get_crew_performance(crew_name, days),
                    "recommendations": [rec.__dict__ for rec in optimization_engine.analyze_crew_performance(crew_name, days)]
                }
            
            with open(export, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            console.print(f"\n[green]✅ Data exported to {export}[/green]")
    
    except Exception as e:
        console.print(f"[red]❌ Error analyzing performance: {str(e)}[/red]")

@app.command()
def logs(
    show_summary: bool = typer.Option(False, "--summary", "-s", help="Show log summary"),
    show_errors: bool = typer.Option(False, "--errors", "-e", help="Show recent errors"),
    show_debug: bool = typer.Option(False, "--debug", "-d", help="Show debug traces"),
    level: str = typer.Option("INFO", "--level", "-l", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    export: Optional[str] = typer.Option(None, "--export", help="Export logs to file"),
    clear: bool = typer.Option(False, "--clear", help="Clear log history")
):
    """Manage logging and debugging.
    
    View logs, debug traces, and error information for troubleshooting.
    """
    try:
        logger = get_logger()
        error_handler = get_error_handler()
        tracer = get_tracer()
        
        if clear:
            # Clear all log data
            error_handler.clear_error_history()
            tracer.clear_traces()
            console.print("[green]✅ Log history cleared[/green]")
            return
        
        if show_summary:
            # Show log summary
            log_summary = logger.get_log_summary()
            error_summary = error_handler.get_error_summary()
            trace_summary = tracer.get_performance_summary()
            
            console.print(f"\n[bold blue]📋 Log Summary[/bold blue]")
            console.print(f"Total Log Entries: {log_summary['total_entries']}")
            console.print(f"Error Count: {error_summary['total_errors']}")
            console.print(f"Trace Events: {trace_summary['total_events']}")
            console.print(f"Error Rate: {error_summary['recovery_success_rate']:.1%}")
            
            if error_summary['errors_by_category']:
                console.print(f"\n[bold]Errors by Category:[/bold]")
                for category, count in error_summary['errors_by_category'].items():
                    console.print(f"  • {category}: {count}")
        
        elif show_errors:
            # Show recent errors
            error_summary = error_handler.get_error_summary()
            console.print(f"\n[bold red]🚨 Recent Errors ({error_summary['total_errors']})[/bold red]")
            
            if error_summary['most_common_errors']:
                console.print(f"\n[bold]Most Common Errors:[/bold]")
                for error_info in error_summary['most_common_errors'][:5]:
                    console.print(f"  • {error_info['error_type']}: {error_info['count']} occurrences")
            
            # Show recent error details
            recent_errors = error_handler.error_history[-10:]  # Last 10 errors
            if recent_errors:
                console.print(f"\n[bold]Recent Error Details:[/bold]")
                for error in recent_errors:
                    console.print(f"\n[red]❌ {error.error_type}[/red]")
                    console.print(f"   Component: {error.component}")
                    console.print(f"   Message: {error.error_message}")
                    console.print(f"   Time: {error.timestamp.strftime('%H:%M:%S')}")
                    if error.recovery_attempted:
                        status = "✅ Recovered" if error.recovery_successful else "❌ Failed"
                        console.print(f"   Recovery: {status}")
        
        elif show_debug:
            # Show debug traces
            trace_summary = tracer.get_performance_summary()
            console.print(f"\n[bold blue]🔍 Debug Traces ({trace_summary['total_events']})[/bold blue]")
            
            if trace_summary['total_events'] > 0:
                console.print(f"Success Rate: {trace_summary['success_rate']:.1%}")
                console.print(f"Average Duration: {trace_summary['average_duration_ms']:.1f}ms")
                console.print(f"Total Duration: {trace_summary['total_duration_ms']:.1f}ms")
                
                # Show slow operations
                slow_ops = tracer.get_slow_operations(1000)  # > 1 second
                if slow_ops:
                    console.print(f"\n[bold]Slow Operations (>1s):[/bold]")
                    for op in slow_ops[:5]:
                        console.print(f"  • {op.component}.{op.function_name}: {op.duration_ms:.1f}ms")
                
                # Show failed operations
                failed_ops = tracer.get_failed_operations()
                if failed_ops:
                    console.print(f"\n[bold]Failed Operations:[/bold]")
                    for op in failed_ops[:5]:
                        console.print(f"  • {op.component}.{op.function_name}: {op.error_message}")
        
        else:
            # Show help
            console.print("\n[bold blue]📋 Log Management[/bold blue]")
            console.print("\n[bold]Available commands:[/bold]")
            console.print("  [cyan]--summary[/cyan] - Show log summary")
            console.print("  [cyan]--errors[/cyan] - Show recent errors")
            console.print("  [cyan]--debug[/cyan] - Show debug traces")
            console.print("  [cyan]--level <level>[/cyan] - Set log level")
            console.print("  [cyan]--export <file>[/cyan] - Export logs to file")
            console.print("  [cyan]--clear[/cyan] - Clear log history")
        
        if export:
            # Export logs
            if export.endswith('.json'):
                # Export as JSON
                export_data = {
                    "log_summary": logger.get_log_summary(),
                    "error_summary": error_handler.get_error_summary(),
                    "trace_summary": tracer.get_performance_summary(),
                    "recent_errors": [
                        {
                            "error_id": e.error_id,
                            "timestamp": e.timestamp.isoformat(),
                            "component": e.component,
                            "error_type": e.error_type,
                            "error_message": e.error_message,
                            "severity": e.severity.value,
                            "category": e.category.value,
                            "recovery_successful": e.recovery_successful
                        }
                        for e in error_handler.error_history[-50:]  # Last 50 errors
                    ]
                }
                
                with open(export, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                # Export as text
                with open(export, 'w') as f:
                    f.write("agentforge Log Export\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Log summary
                    log_summary = logger.get_log_summary()
                    f.write(f"Log Summary:\n")
                    f.write(f"Total Entries: {log_summary['total_entries']}\n")
                    f.write(f"Error Count: {log_summary['error_count']}\n\n")
                    
                    # Recent errors
                    recent_errors = error_handler.error_history[-20:]
                    f.write("Recent Errors:\n")
                    for error in recent_errors:
                        f.write(f"- {error.timestamp}: {error.error_type} in {error.component}\n")
                        f.write(f"  {error.error_message}\n\n")
            
            console.print(f"\n[green]✅ Logs exported to {export}[/green]")
    
    except Exception as e:
        console.print(f"[red]❌ Error managing logs: {str(e)}[/red]")

@app.command()
def adaptive(
    action: str = typer.Argument(..., help="Action to perform: analyze, create, insights, learn"),
    crew_name: Optional[str] = typer.Option(None, "--crew", "-c", help="Crew name for analysis"),
    task_context: Optional[str] = typer.Option(None, "--context", help="Task context for agent creation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """🧠 Manage adaptive agent creation and reinforcement learning.
    
    The adaptive system can automatically create new agents based on performance
    feedback and task requirements using reinforcement learning principles.
    
    Examples:
      agentforge adaptive analyze --crew my_crew
      agentforge adaptive create --crew my_crew --context "complex data analysis"
      agentforge adaptive insights
      agentforge adaptive learn --feedback "performance_improved"
    """
    try:
        config = Config()
        adaptive_manager = AdaptiveAgentManager(config)
        
        if action == "analyze":
            if not crew_name:
                console.print("[red]❌ Crew name required for analysis[/red]")
                raise typer.Exit(1)
            
            console.print(f"[bold green]🧠 Analyzing adaptive opportunities for crew:[/bold green] {crew_name}")
            
            analysis = adaptive_manager.analyze_agent_performance(crew_name)
            
            if analysis["status"] == "success":
                console.print(f"\n[bold blue]📊 Performance Analysis:[/bold blue]")
                console.print(f"  Overall Score: {analysis['overall_performance']['overall_score']:.2f}")
                console.print(f"  Agent Count: {analysis['overall_performance']['agent_count']}")
                
                if analysis["recommendations"]:
                    console.print(f"\n[bold yellow]💡 Recommendations:[/bold yellow]")
                    for i, rec in enumerate(analysis["recommendations"], 1):
                        console.print(f"  {i}. {rec['suggestion']}")
                        console.print(f"     Issue: {rec['issue']} (Current: {rec['current_value']:.2f}, Threshold: {rec['threshold']:.2f})")
                else:
                    console.print(f"\n[green]✅ No immediate improvements needed[/green]")
            else:
                console.print(f"[red]❌ Analysis failed: {analysis.get('message', 'Unknown error')}[/red]")
        
        elif action == "create":
            if not crew_name:
                console.print("[red]❌ Crew name required for agent creation[/red]")
                raise typer.Exit(1)
            
            console.print(f"[bold green]🧠 Evaluating agent creation for crew:[/bold green] {crew_name}")
            
            # Parse task context if provided
            context = {}
            if task_context:
                try:
                    context = json.loads(task_context)
                except json.JSONDecodeError:
                    context = {"description": task_context, "complexity": 0.7}
            
            decision = adaptive_manager.decide_agent_creation(crew_name, context)
            
            if decision:
                console.print(f"\n[bold yellow]🎯 Agent Creation Decision:[/bold yellow]")
                console.print(f"  Trigger: {decision.trigger.value}")
                console.print(f"  Reasoning: {decision.reasoning}")
                console.print(f"  Confidence: {decision.confidence:.2f}")
                console.print(f"  Expected Improvement: {decision.expected_improvement:.2f}")
                console.print(f"  Creation Cost: ${decision.creation_cost:.2f}")
                
                if typer.confirm("\n🤖 Create this adaptive agent?"):
                    result = adaptive_manager.create_adaptive_agent(decision)
                    
                    if result["status"] == "success":
                        console.print(f"\n[bold green]✅ Adaptive agent created successfully![/bold green]")
                        console.print(f"  Agent Name: {result['agent_name']}")
                        console.print(f"  Specialization: {result['specialization']}")
                        console.print(f"  Crew Path: {result['crew_path']}")
                        console.print(f"  Expected Improvement: {result['expected_improvement']:.2f}")
                    else:
                        console.print(f"[red]❌ Failed to create agent: {result.get('message', 'Unknown error')}[/red]")
                else:
                    console.print("[yellow]⏸️  Agent creation cancelled[/yellow]")
            else:
                console.print(f"\n[green]✅ No agent creation needed at this time[/green]")
                console.print("The current crew performance is within acceptable parameters.")
        
        elif action == "insights":
            console.print(f"[bold green]🧠 Adaptive System Insights:[/bold green]")
            
            insights = adaptive_manager.get_adaptive_insights()
            
            console.print(f"\n[bold blue]📈 System Statistics:[/bold blue]")
            console.print(f"  Total Agents Created: {insights['total_agents_created']}")
            console.print(f"  Learning Rate: {insights['current_learning_rate']:.3f}")
            console.print(f"  Exploration Rate: {insights['current_exploration_rate']:.3f}")
            console.print(f"  Average Confidence: {insights['average_confidence']:.2f}")
            console.print(f"  Performance Trend: {insights['performance_trend']}")
            
            if insights['recent_triggers']:
                console.print(f"\n[bold blue]🔄 Recent Triggers:[/bold blue]")
                for trigger in insights['recent_triggers']:
                    console.print(f"  • {trigger}")
        
        elif action == "learn":
            feedback = typer.prompt("Enter feedback (performance_improved/performance_degraded/neutral)")
            
            feedback_data = {
                "performance_improved": feedback == "performance_improved",
                "performance_degraded": feedback == "performance_degraded",
                "timestamp": time.time()
            }
            
            adaptive_manager.update_learning_parameters(feedback_data)
            console.print(f"[green]✅ Learning parameters updated based on feedback: {feedback}[/green]")
        
        else:
            console.print(f"[red]❌ Unknown action: {action}[/red]")
            console.print("Available actions: analyze, create, insights, learn")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[red]❌ Error in adaptive system: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def rl(
    action: str = typer.Argument(..., help="Action to perform: step, train, insights, reset"),
    crew_name: Optional[str] = typer.Option(None, "--crew", "-c", help="Crew name for RL operations"),
    task_context: Optional[str] = typer.Option(None, "--context", help="Task context for RL step"),
    episodes: int = typer.Option(10, "--episodes", "-e", help="Number of training episodes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """🤖 Reinforcement Learning agent creation and training.
    
    The RL system learns when and how to create new agents based on performance
    feedback using Q-learning and experience replay.
    
    Examples:
      agentforge rl step --crew my_crew --context "complex analysis task"
      agentforge rl train --crew my_crew --episodes 50
      agentforge rl insights
      agentforge rl reset
    """
    try:
        config = Config()
        rl_creator = RLAgentCreator(config)
        
        if action == "step":
            if not crew_name:
                console.print("[red]❌ Crew name required for RL step[/red]")
                raise typer.Exit(1)
            
            console.print(f"[bold green]🤖 Executing RL step for crew:[/bold green] {crew_name}")
            
            # Parse task context if provided
            context = {}
            if task_context:
                try:
                    context = json.loads(task_context)
                except json.JSONDecodeError:
                    context = {"description": task_context, "complexity": 0.7}
            
            result = rl_creator.step(crew_name, context)
            
            if result["status"] == "success":
                console.print(f"\n[bold blue]🧠 RL Step Results:[/bold blue]")
                console.print(f"  Current State: {result['state']}")
                console.print(f"  Action Taken: {result['action']}")
                console.print(f"  Next State: {result['next_state']}")
                console.print(f"  Reward: {result['reward']:.3f}")
                console.print(f"  Episode Reward: {result['episode_reward']:.3f}")
                
                if result['result']['status'] == 'agent_created':
                    console.print(f"\n[bold green]✅ New Agent Created:[/bold green]")
                    console.print(f"  Agent Name: {result['result']['agent_name']}")
                    console.print(f"  Specialization: {result['result']['specialization']}")
            else:
                console.print(f"[red]❌ RL step failed: {result.get('message', 'Unknown error')}[/red]")
        
        elif action == "train":
            if not crew_name:
                console.print("[red]❌ Crew name required for training[/red]")
                raise typer.Exit(1)
            
            console.print(f"[bold green]🤖 Training RL agent for {episodes} episodes...[/bold green]")
            
            # Training loop
            for episode in range(episodes):
                console.print(f"\n[dim]Episode {episode + 1}/{episodes}[/dim]")
                
                # Generate random task context for training
                context = {
                    "description": f"Training task {episode + 1}",
                    "complexity": np.random.uniform(0.3, 0.9),
                    "required_capabilities": np.random.choice(
                        ["analysis", "creative", "technical", "communication"], 
                        size=np.random.randint(1, 4), 
                        replace=False
                    ).tolist()
                }
                
                result = rl_creator.step(crew_name, context)
                
                if verbose and result["status"] == "success":
                    console.print(f"  State: {result['state']} -> Action: {result['action']} -> Reward: {result['reward']:.3f}")
                
                # End episode every 10 steps
                if (episode + 1) % 10 == 0:
                    rl_creator.end_episode()
                    console.print(f"  [dim]Episode {episode + 1} completed[/dim]")
            
            # Final episode end
            rl_creator.end_episode()
            console.print(f"\n[bold green]✅ Training completed![/bold green]")
        
        elif action == "insights":
            console.print(f"[bold green]🤖 RL System Insights:[/bold green]")
            
            insights = rl_creator.get_rl_insights()
            rl_stats = insights["rl_agent_stats"]
            
            if rl_stats["status"] == "no_data":
                console.print("[yellow]⚠️  No training data available[/yellow]")
                return
            
            console.print(f"\n[bold blue]📊 Learning Statistics:[/bold blue]")
            console.print(f"  Total Episodes: {rl_stats['total_episodes']}")
            console.print(f"  Average Reward: {rl_stats['average_reward']:.3f}")
            console.print(f"  Average Episode Length: {rl_stats['average_episode_length']:.1f}")
            console.print(f"  Exploration Rate: {rl_stats['exploration_rate']:.3f}")
            console.print(f"  Q-Table Size: {rl_stats['q_table_size']}")
            console.print(f"  Learning Progress: {rl_stats['learning_progress']}")
            
            console.print(f"\n[bold blue]🎯 Current State:[/bold blue]")
            console.print(f"  State: {insights['current_state']}")
            console.print(f"  Episode Reward: {insights['episode_reward']:.3f}")
            console.print(f"  Episode Actions: {insights['episode_actions']}")
            console.print(f"  Model Path: {insights['model_path']}")
        
        elif action == "reset":
            if typer.confirm("Are you sure you want to reset the RL model? This will delete all learning progress."):
                # Reset RL agent
                rl_creator.rl_agent = QLearningAgent()
                rl_creator.episode_reward = 0.0
                rl_creator.episode_actions = []
                rl_creator.current_state = RLState.NORMAL_OPERATION
                
                # Delete model file
                if rl_creator.model_path.exists():
                    rl_creator.model_path.unlink()
                
                console.print("[green]✅ RL model reset successfully[/green]")
            else:
                console.print("[yellow]⏸️  Reset cancelled[/yellow]")
        
        else:
            console.print(f"[red]❌ Unknown action: {action}[/red]")
            console.print("Available actions: step, train, insights, reset")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[red]❌ Error in RL system: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def version():
    """Show AgentForge version."""
    try:
        from . import __version__
        console.print(f"[bold green]🔥 AgentForge[/bold green] version [cyan]{__version__}[/cyan]")
        console.print(f"[dim]⚡ Forge intelligent AI agents with CrewAI[/dim]")
    except ImportError:
        console.print(f"[bold green]🔥 AgentForge[/bold green] version [cyan]0.2.0[/cyan]")
        console.print(f"[dim]⚡ Forge intelligent AI agents with CrewAI[/dim]")

def main():
    """Main CLI entry point."""
    app()

if __name__ == "__main__":
    main()