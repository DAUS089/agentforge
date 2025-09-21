"""
Crew Template Library for agentforge.

Provides pre-built crew patterns for common use cases.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class WorkflowType(Enum):
    """Types of crew workflows."""
    SEQUENTIAL = "sequential"
    COLLABORATIVE = "collaborative"
    HIERARCHICAL = "hierarchical"
    PARALLEL = "parallel"


@dataclass
class AgentTemplate:
    """Template for an agent within a crew."""
    name: str
    role: str
    goal: str
    backstory: str
    tools: List[str]
    memory_type: str = "conversation_buffer"
    max_iter: int = 5
    verbose: bool = True


@dataclass
class TaskTemplate:
    """Template for a task within a crew."""
    name: str
    description: str
    expected_output: str
    context: Optional[List[str]] = None
    tools: Optional[List[str]] = None


@dataclass
class CrewTemplate:
    """Template for a complete crew."""
    name: str
    description: str
    category: str
    workflow: WorkflowType
    agents: List[AgentTemplate]
    tasks: List[TaskTemplate]
    tools: List[str]
    estimated_duration: str
    complexity: str  # "low", "medium", "high"
    use_cases: List[str]


class CrewTemplateLibrary:
    """Library of pre-built crew templates."""
    
    def __init__(self):
        self._templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, CrewTemplate]:
        """Initialize all available crew templates."""
        return {
            "data_analysis": self._create_data_analysis_template(),
            "web_scraping": self._create_web_scraping_template(),
            "content_creation": self._create_content_creation_template(),
            "code_review": self._create_code_review_template(),
            "research_crew": self._create_research_template(),
            "customer_support": self._create_customer_support_template(),
            "marketing_automation": self._create_marketing_template(),
            "financial_analysis": self._create_financial_analysis_template(),
            "bug_triage": self._create_bug_triage_template(),
            "documentation": self._create_documentation_template()
        }
    
    def get_template(self, template_name: str) -> Optional[CrewTemplate]:
        """Get a crew template by name."""
        return self._templates.get(template_name.lower())
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self._templates.keys())
    
    def get_templates_by_category(self, category: str) -> List[CrewTemplate]:
        """Get all templates in a specific category."""
        return [template for template in self._templates.values() 
                if template.category.lower() == category.lower()]
    
    def search_templates(self, query: str) -> List[CrewTemplate]:
        """Search templates by name, description, or use case."""
        query = query.lower()
        results = []
        
        for template in self._templates.values():
            if (query in template.name.lower() or 
                query in template.description.lower() or
                any(query in use_case.lower() for use_case in template.use_cases)):
                results.append(template)
        
        return results
    
    def _create_data_analysis_template(self) -> CrewTemplate:
        """Create data analysis crew template."""
        return CrewTemplate(
            name="Data Analysis Crew",
            description="A crew specialized in data collection, analysis, and reporting",
            category="analytics",
            workflow=WorkflowType.SEQUENTIAL,
            agents=[
                AgentTemplate(
                    name="DataCollector",
                    role="Data Collection Specialist",
                    goal="Gather and organize data from various sources",
                    backstory="Expert in data collection with experience in APIs, web scraping, and database queries",
                    tools=["CSVSearchTool", "DirectoryReadTool", "WebSearchTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="DataAnalyzer",
                    role="Data Analysis Expert",
                    goal="Analyze data to extract meaningful insights and patterns",
                    backstory="Data scientist with expertise in statistical analysis and machine learning",
                    tools=["CodeInterpreterTool", "FileReadTool", "CSVSearchTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="ReportGenerator",
                    role="Report Writer",
                    goal="Create comprehensive reports and visualizations from analysis results",
                    backstory="Technical writer with experience in data visualization and business reporting",
                    tools=["FileWriteTool", "DirectoryReadTool"],
                    memory_type="conversation_buffer"
                )
            ],
            tasks=[
                TaskTemplate(
                    name="collect_data",
                    description="Collect data from specified sources",
                    expected_output="Raw data files and summary of data sources",
                    tools=["CSVSearchTool", "DirectoryReadTool", "WebSearchTool"]
                ),
                TaskTemplate(
                    name="analyze_data",
                    description="Perform statistical analysis on collected data",
                    expected_output="Analysis results with key findings and patterns",
                    context=["collect_data"],
                    tools=["CodeInterpreterTool", "FileReadTool"]
                ),
                TaskTemplate(
                    name="generate_report",
                    description="Create comprehensive report with findings and recommendations",
                    expected_output="Final report document with analysis and recommendations",
                    context=["analyze_data"],
                    tools=["FileWriteTool", "DirectoryReadTool"]
                )
            ],
            tools=["CSVSearchTool", "DirectoryReadTool", "WebSearchTool", "CodeInterpreterTool", "FileReadTool", "FileWriteTool"],
            estimated_duration="2-4 hours",
            complexity="medium",
            use_cases=["Business intelligence", "Market research", "Performance analysis", "Trend analysis"]
        )
    
    def _create_web_scraping_template(self) -> CrewTemplate:
        """Create web scraping crew template."""
        return CrewTemplate(
            name="Web Scraping Crew",
            description="A crew specialized in web scraping and data extraction",
            category="automation",
            workflow=WorkflowType.SEQUENTIAL,
            agents=[
                AgentTemplate(
                    name="URLCollector",
                    role="URL Discovery Specialist",
                    goal="Find and collect relevant URLs for scraping",
                    backstory="Expert in web crawling and URL discovery techniques",
                    tools=["WebSearchTool", "DirectoryReadTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="ContentExtractor",
                    role="Web Scraping Expert",
                    goal="Extract content from web pages efficiently and ethically",
                    backstory="Web scraping specialist with expertise in various extraction techniques",
                    tools=["ScrapeWebsiteTool", "WebsiteSearchTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="DataProcessor",
                    role="Data Processing Specialist",
                    goal="Clean, structure, and organize scraped data",
                    backstory="Data processing expert with experience in data cleaning and normalization",
                    tools=["FileWriteTool", "FileReadTool", "CSVSearchTool"],
                    memory_type="conversation_buffer"
                )
            ],
            tasks=[
                TaskTemplate(
                    name="discover_urls",
                    description="Find and collect relevant URLs for scraping",
                    expected_output="List of URLs to scrape with metadata",
                    tools=["WebSearchTool", "DirectoryReadTool"]
                ),
                TaskTemplate(
                    name="extract_content",
                    description="Scrape content from discovered URLs",
                    expected_output="Raw scraped content from all URLs",
                    context=["discover_urls"],
                    tools=["ScrapeWebsiteTool", "WebsiteSearchTool"]
                ),
                TaskTemplate(
                    name="process_data",
                    description="Clean and structure the scraped data",
                    expected_output="Clean, structured data in appropriate format",
                    context=["extract_content"],
                    tools=["FileWriteTool", "FileReadTool", "CSVSearchTool"]
                )
            ],
            tools=["WebSearchTool", "DirectoryReadTool", "ScrapeWebsiteTool", "WebsiteSearchTool", "FileWriteTool", "FileReadTool", "CSVSearchTool"],
            estimated_duration="1-3 hours",
            complexity="medium",
            use_cases=["Competitor analysis", "Price monitoring", "Content aggregation", "Lead generation"]
        )
    
    def _create_content_creation_template(self) -> CrewTemplate:
        """Create content creation crew template."""
        return CrewTemplate(
            name="Content Creation Crew",
            description="A crew specialized in creating high-quality content",
            category="content",
            workflow=WorkflowType.COLLABORATIVE,
            agents=[
                AgentTemplate(
                    name="Researcher",
                    role="Content Researcher",
                    goal="Research topics and gather information for content creation",
                    backstory="Research specialist with expertise in fact-checking and information gathering",
                    tools=["WebSearchTool", "DirectoryReadTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="Writer",
                    role="Content Writer",
                    goal="Create engaging and well-structured content",
                    backstory="Professional writer with experience in various content types and styles",
                    tools=["FileWriteTool", "FileReadTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="Editor",
                    role="Content Editor",
                    goal="Review, edit, and polish content for quality and consistency",
                    backstory="Editor with expertise in grammar, style, and content optimization",
                    tools=["FileReadTool", "FileWriteTool"],
                    memory_type="conversation_buffer"
                )
            ],
            tasks=[
                TaskTemplate(
                    name="research_topic",
                    description="Research the topic and gather relevant information",
                    expected_output="Comprehensive research notes and sources",
                    tools=["WebSearchTool", "DirectoryReadTool"]
                ),
                TaskTemplate(
                    name="create_content",
                    description="Write the main content based on research",
                    expected_output="Draft content ready for review",
                    context=["research_topic"],
                    tools=["FileWriteTool", "FileReadTool"]
                ),
                TaskTemplate(
                    name="edit_content",
                    description="Review and edit the content for quality",
                    expected_output="Final polished content",
                    context=["create_content"],
                    tools=["FileReadTool", "FileWriteTool"]
                )
            ],
            tools=["WebSearchTool", "DirectoryReadTool", "FileWriteTool", "FileReadTool"],
            estimated_duration="2-6 hours",
            complexity="medium",
            use_cases=["Blog posts", "Articles", "Documentation", "Marketing copy", "Technical writing"]
        )
    
    def _create_code_review_template(self) -> CrewTemplate:
        """Create code review crew template."""
        return CrewTemplate(
            name="Code Review Crew",
            description="A crew specialized in code review and quality assurance",
            category="development",
            workflow=WorkflowType.PARALLEL,
            agents=[
                AgentTemplate(
                    name="CodeAnalyzer",
                    role="Code Analysis Expert",
                    goal="Analyze code for bugs, performance issues, and best practices",
                    backstory="Senior developer with expertise in code analysis and debugging",
                    tools=["CodeInterpreterTool", "FileReadTool", "GithubSearchTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="SecurityAuditor",
                    role="Security Specialist",
                    goal="Identify security vulnerabilities and compliance issues",
                    backstory="Security expert with experience in vulnerability assessment",
                    tools=["FileReadTool", "CodeInterpreterTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="DocumentationReviewer",
                    role="Documentation Specialist",
                    goal="Review and improve code documentation and comments",
                    backstory="Technical writer with expertise in code documentation",
                    tools=["FileReadTool", "FileWriteTool"],
                    memory_type="conversation_buffer"
                )
            ],
            tasks=[
                TaskTemplate(
                    name="analyze_code",
                    description="Analyze code for bugs, performance, and best practices",
                    expected_output="Code analysis report with findings and recommendations",
                    tools=["CodeInterpreterTool", "FileReadTool", "GithubSearchTool"]
                ),
                TaskTemplate(
                    name="security_audit",
                    description="Perform security audit of the code",
                    expected_output="Security audit report with vulnerabilities and fixes",
                    tools=["FileReadTool", "CodeInterpreterTool"]
                ),
                TaskTemplate(
                    name="review_documentation",
                    description="Review and improve code documentation",
                    expected_output="Documentation review with improvements",
                    tools=["FileReadTool", "FileWriteTool"]
                )
            ],
            tools=["CodeInterpreterTool", "FileReadTool", "GithubSearchTool", "FileWriteTool"],
            estimated_duration="1-2 hours",
            complexity="high",
            use_cases=["Pull request review", "Code quality audit", "Security assessment", "Documentation review"]
        )
    
    def _create_research_template(self) -> CrewTemplate:
        """Create research crew template."""
        return CrewTemplate(
            name="Research Crew",
            description="A crew specialized in comprehensive research and analysis",
            category="research",
            workflow=WorkflowType.HIERARCHICAL,
            agents=[
                AgentTemplate(
                    name="ResearchCoordinator",
                    role="Research Coordinator",
                    goal="Coordinate research activities and synthesize findings",
                    backstory="Research manager with expertise in project coordination and synthesis",
                    tools=["WebSearchTool", "FileWriteTool", "DirectoryReadTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="DataResearcher",
                    role="Data Research Specialist",
                    goal="Gather quantitative data and statistics",
                    backstory="Data researcher with expertise in statistical analysis and data collection",
                    tools=["CSVSearchTool", "WebSearchTool", "CodeInterpreterTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="LiteratureResearcher",
                    role="Literature Research Specialist",
                    goal="Research academic papers, articles, and publications",
                    backstory="Academic researcher with expertise in literature review and analysis",
                    tools=["WebSearchTool", "FileReadTool", "DirectoryReadTool"],
                    memory_type="conversation_buffer"
                )
            ],
            tasks=[
                TaskTemplate(
                    name="coordinate_research",
                    description="Plan and coordinate research activities",
                    expected_output="Research plan and coordination strategy",
                    tools=["WebSearchTool", "FileWriteTool", "DirectoryReadTool"]
                ),
                TaskTemplate(
                    name="gather_data",
                    description="Collect quantitative data and statistics",
                    expected_output="Data collection report with sources and methodology",
                    context=["coordinate_research"],
                    tools=["CSVSearchTool", "WebSearchTool", "CodeInterpreterTool"]
                ),
                TaskTemplate(
                    name="literature_review",
                    description="Research academic literature and publications",
                    expected_output="Literature review with key findings and citations",
                    context=["coordinate_research"],
                    tools=["WebSearchTool", "FileReadTool", "DirectoryReadTool"]
                )
            ],
            tools=["WebSearchTool", "FileWriteTool", "DirectoryReadTool", "CSVSearchTool", "CodeInterpreterTool", "FileReadTool"],
            estimated_duration="4-8 hours",
            complexity="high",
            use_cases=["Academic research", "Market analysis", "Competitive intelligence", "Trend analysis"]
        )
    
    def _create_customer_support_template(self) -> CrewTemplate:
        """Create customer support crew template."""
        return CrewTemplate(
            name="Customer Support Crew",
            description="A crew specialized in customer support and issue resolution",
            category="support",
            workflow=WorkflowType.SEQUENTIAL,
            agents=[
                AgentTemplate(
                    name="TicketTriage",
                    role="Support Ticket Triage Specialist",
                    goal="Categorize and prioritize customer support tickets",
                    backstory="Customer service expert with experience in ticket management and prioritization",
                    tools=["FileReadTool", "DirectoryReadTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="TechnicalResolver",
                    role="Technical Support Specialist",
                    goal="Resolve technical issues and provide solutions",
                    backstory="Technical support expert with deep product knowledge and troubleshooting skills",
                    tools=["CodeInterpreterTool", "FileReadTool", "WebSearchTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="ResponseWriter",
                    role="Response Writer",
                    goal="Write clear and helpful responses to customers",
                    backstory="Technical writer with expertise in customer communication",
                    tools=["FileWriteTool", "FileReadTool"],
                    memory_type="conversation_buffer"
                )
            ],
            tasks=[
                TaskTemplate(
                    name="triage_tickets",
                    description="Categorize and prioritize support tickets",
                    expected_output="Categorized tickets with priority levels",
                    tools=["FileReadTool", "DirectoryReadTool"]
                ),
                TaskTemplate(
                    name="resolve_issues",
                    description="Research and resolve technical issues",
                    expected_output="Solutions and troubleshooting steps for each issue",
                    context=["triage_tickets"],
                    tools=["CodeInterpreterTool", "FileReadTool", "WebSearchTool"]
                ),
                TaskTemplate(
                    name="write_responses",
                    description="Write customer responses with solutions",
                    expected_output="Professional customer responses with solutions",
                    context=["resolve_issues"],
                    tools=["FileWriteTool", "FileReadTool"]
                )
            ],
            tools=["FileReadTool", "DirectoryReadTool", "CodeInterpreterTool", "WebSearchTool", "FileWriteTool"],
            estimated_duration="1-3 hours",
            complexity="medium",
            use_cases=["Ticket management", "Issue resolution", "Customer communication", "Technical support"]
        )
    
    def _create_marketing_template(self) -> CrewTemplate:
        """Create marketing automation crew template."""
        return CrewTemplate(
            name="Marketing Automation Crew",
            description="A crew specialized in marketing content and campaign automation",
            category="marketing",
            workflow=WorkflowType.COLLABORATIVE,
            agents=[
                AgentTemplate(
                    name="MarketResearcher",
                    role="Market Research Specialist",
                    goal="Research market trends and competitor strategies",
                    backstory="Marketing researcher with expertise in market analysis and competitive intelligence",
                    tools=["WebSearchTool", "CSVSearchTool", "DirectoryReadTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="ContentCreator",
                    role="Marketing Content Creator",
                    goal="Create engaging marketing content and campaigns",
                    backstory="Marketing content creator with expertise in various content formats and channels",
                    tools=["FileWriteTool", "FileReadTool", "WebSearchTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="CampaignAnalyst",
                    role="Campaign Performance Analyst",
                    goal="Analyze campaign performance and optimize strategies",
                    backstory="Marketing analyst with expertise in campaign optimization and performance measurement",
                    tools=["CodeInterpreterTool", "CSVSearchTool", "FileReadTool"],
                    memory_type="conversation_buffer"
                )
            ],
            tasks=[
                TaskTemplate(
                    name="market_research",
                    description="Research market trends and competitor strategies",
                    expected_output="Market research report with insights and recommendations",
                    tools=["WebSearchTool", "CSVSearchTool", "DirectoryReadTool"]
                ),
                TaskTemplate(
                    name="create_content",
                    description="Create marketing content and campaign materials",
                    expected_output="Marketing content and campaign materials",
                    context=["market_research"],
                    tools=["FileWriteTool", "FileReadTool", "WebSearchTool"]
                ),
                TaskTemplate(
                    name="analyze_performance",
                    description="Analyze campaign performance and provide optimization recommendations",
                    expected_output="Performance analysis report with optimization recommendations",
                    context=["create_content"],
                    tools=["CodeInterpreterTool", "CSVSearchTool", "FileReadTool"]
                )
            ],
            tools=["WebSearchTool", "CSVSearchTool", "DirectoryReadTool", "FileWriteTool", "FileReadTool", "CodeInterpreterTool"],
            estimated_duration="3-6 hours",
            complexity="medium",
            use_cases=["Campaign creation", "Content marketing", "Market analysis", "Performance optimization"]
        )
    
    def _create_financial_analysis_template(self) -> CrewTemplate:
        """Create financial analysis crew template."""
        return CrewTemplate(
            name="Financial Analysis Crew",
            description="A crew specialized in financial analysis and reporting",
            category="finance",
            workflow=WorkflowType.SEQUENTIAL,
            agents=[
                AgentTemplate(
                    name="DataCollector",
                    role="Financial Data Collector",
                    goal="Collect financial data from various sources",
                    backstory="Financial data specialist with expertise in data collection and validation",
                    tools=["CSVSearchTool", "WebSearchTool", "FileReadTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="FinancialAnalyst",
                    role="Financial Analyst",
                    goal="Analyze financial data and generate insights",
                    backstory="Financial analyst with expertise in financial modeling and analysis",
                    tools=["CodeInterpreterTool", "FileReadTool", "CSVSearchTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="ReportGenerator",
                    role="Financial Report Writer",
                    goal="Create comprehensive financial reports and recommendations",
                    backstory="Financial writer with expertise in financial reporting and communication",
                    tools=["FileWriteTool", "FileReadTool"],
                    memory_type="conversation_buffer"
                )
            ],
            tasks=[
                TaskTemplate(
                    name="collect_financial_data",
                    description="Collect financial data from various sources",
                    expected_output="Financial data files and summary of sources",
                    tools=["CSVSearchTool", "WebSearchTool", "FileReadTool"]
                ),
                TaskTemplate(
                    name="analyze_financials",
                    description="Perform financial analysis and modeling",
                    expected_output="Financial analysis with key metrics and insights",
                    context=["collect_financial_data"],
                    tools=["CodeInterpreterTool", "FileReadTool", "CSVSearchTool"]
                ),
                TaskTemplate(
                    name="generate_financial_report",
                    description="Create comprehensive financial report",
                    expected_output="Financial report with analysis and recommendations",
                    context=["analyze_financials"],
                    tools=["FileWriteTool", "FileReadTool"]
                )
            ],
            tools=["CSVSearchTool", "WebSearchTool", "FileReadTool", "CodeInterpreterTool", "FileWriteTool"],
            estimated_duration="2-4 hours",
            complexity="high",
            use_cases=["Financial reporting", "Investment analysis", "Budget planning", "Risk assessment"]
        )
    
    def _create_bug_triage_template(self) -> CrewTemplate:
        """Create bug triage crew template."""
        return CrewTemplate(
            name="Bug Triage Crew",
            description="A crew specialized in bug triage and issue management",
            category="development",
            workflow=WorkflowType.HIERARCHICAL,
            agents=[
                AgentTemplate(
                    name="BugTriage",
                    role="Bug Triage Specialist",
                    goal="Categorize and prioritize bug reports",
                    backstory="QA specialist with expertise in bug classification and prioritization",
                    tools=["FileReadTool", "DirectoryReadTool", "GithubSearchTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="BugReproducer",
                    role="Bug Reproduction Specialist",
                    goal="Reproduce and verify bug reports",
                    backstory="QA engineer with expertise in bug reproduction and testing",
                    tools=["CodeInterpreterTool", "FileReadTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="BugResolver",
                    role="Bug Resolution Specialist",
                    goal="Investigate and provide solutions for bugs",
                    backstory="Senior developer with expertise in debugging and problem-solving",
                    tools=["CodeInterpreterTool", "FileReadTool", "GithubSearchTool"],
                    memory_type="conversation_buffer"
                )
            ],
            tasks=[
                TaskTemplate(
                    name="triage_bugs",
                    description="Categorize and prioritize bug reports",
                    expected_output="Categorized bug reports with priority levels",
                    tools=["FileReadTool", "DirectoryReadTool", "GithubSearchTool"]
                ),
                TaskTemplate(
                    name="reproduce_bugs",
                    description="Reproduce and verify reported bugs",
                    expected_output="Bug reproduction steps and verification results",
                    context=["triage_bugs"],
                    tools=["CodeInterpreterTool", "FileReadTool"]
                ),
                TaskTemplate(
                    name="resolve_bugs",
                    description="Investigate and provide solutions for bugs",
                    expected_output="Bug solutions and fixes",
                    context=["reproduce_bugs"],
                    tools=["CodeInterpreterTool", "FileReadTool", "GithubSearchTool"]
                )
            ],
            tools=["FileReadTool", "DirectoryReadTool", "GithubSearchTool", "CodeInterpreterTool"],
            estimated_duration="1-3 hours",
            complexity="medium",
            use_cases=["Bug management", "Issue tracking", "Quality assurance", "Problem resolution"]
        )
    
    def _create_documentation_template(self) -> CrewTemplate:
        """Create documentation crew template."""
        return CrewTemplate(
            name="Documentation Crew",
            description="A crew specialized in creating and maintaining documentation",
            category="documentation",
            workflow=WorkflowType.COLLABORATIVE,
            agents=[
                AgentTemplate(
                    name="DocumentationAnalyzer",
                    role="Documentation Analyst",
                    goal="Analyze existing documentation and identify gaps",
                    backstory="Technical writer with expertise in documentation analysis and gap identification",
                    tools=["FileReadTool", "DirectoryReadTool", "WebSearchTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="ContentWriter",
                    role="Technical Writer",
                    goal="Write clear and comprehensive documentation",
                    backstory="Technical writer with expertise in various documentation formats and styles",
                    tools=["FileWriteTool", "FileReadTool"],
                    memory_type="conversation_buffer"
                ),
                AgentTemplate(
                    name="DocumentationReviewer",
                    role="Documentation Reviewer",
                    goal="Review and improve documentation quality",
                    backstory="Senior technical writer with expertise in documentation review and improvement",
                    tools=["FileReadTool", "FileWriteTool"],
                    memory_type="conversation_buffer"
                )
            ],
            tasks=[
                TaskTemplate(
                    name="analyze_documentation",
                    description="Analyze existing documentation and identify gaps",
                    expected_output="Documentation analysis report with identified gaps",
                    tools=["FileReadTool", "DirectoryReadTool", "WebSearchTool"]
                ),
                TaskTemplate(
                    name="write_documentation",
                    description="Write new documentation content",
                    expected_output="New documentation content",
                    context=["analyze_documentation"],
                    tools=["FileWriteTool", "FileReadTool"]
                ),
                TaskTemplate(
                    name="review_documentation",
                    description="Review and improve documentation quality",
                    expected_output="Improved documentation with quality enhancements",
                    context=["write_documentation"],
                    tools=["FileReadTool", "FileWriteTool"]
                )
            ],
            tools=["FileReadTool", "DirectoryReadTool", "WebSearchTool", "FileWriteTool"],
            estimated_duration="2-4 hours",
            complexity="medium",
            use_cases=["API documentation", "User guides", "Technical specifications", "Process documentation"]
        )
