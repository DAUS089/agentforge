"""
Custom tools for tech_blog_writer_final crew.

This module handles tool initialization and provides tools to agents.
"""

from typing import List, Any


def get_tools_for_agent(tool_names: List[str]) -> List[Any]:
    """Get actual CrewAI tools for an agent based on tool names."""
    # Try to import available tools
    available_tools = {}
    
    try:
        from crewai_tools import (
            WebsiteSearchTool, SerperDevTool, FileReadTool, ScrapeWebsiteTool, GithubSearchTool,
            YoutubeVideoSearchTool, YoutubeChannelSearchTool, CodeInterpreterTool,
            PDFSearchTool, DOCXSearchTool, CSVSearchTool, JSONSearchTool,
            XMLSearchTool, TXTSearchTool, MDXSearchTool, DirectoryReadTool,
            DirectorySearchTool
        )
        
        available_tools = {
            'SerperDevTool': SerperDevTool,
            'FileReadTool': FileReadTool,
            'ScrapeWebsiteTool': ScrapeWebsiteTool,
            'GithubSearchTool': GithubSearchTool,
            'YoutubeVideoSearchTool': YoutubeVideoSearchTool,
            'YoutubeChannelSearchTool': YoutubeChannelSearchTool,
            'CodeInterpreterTool': CodeInterpreterTool,
            'PDFSearchTool': PDFSearchTool,
            'DOCXSearchTool': DOCXSearchTool,
            'CSVSearchTool': CSVSearchTool,
            'JSONSearchTool': JSONSearchTool,
            'XMLSearchTool': XMLSearchTool,
            'TXTSearchTool': TXTSearchTool,
            'MDXSearchTool': MDXSearchTool,
            'DirectoryReadTool': DirectoryReadTool,
            'DirectorySearchTool': DirectorySearchTool,
            'WebsiteSearchTool': WebsiteSearchTool
        }
        
    except ImportError:
        print("Warning: crewai-tools not installed, using mock tools")
        return []
    
    tools = []
    
    for tool_name in tool_names:
        try:
            if tool_name in available_tools:
                tool_class = available_tools[tool_name]
                tools.append(tool_class())
            else:
                print(f"Warning: Unknown tool '{tool_name}', using SerperDevTool as fallback")
                if 'SerperDevTool' in available_tools and not any(type(t).__name__ == 'SerperDevTool' for t in tools):
                    tools.append(available_tools['SerperDevTool']())
        except Exception as e:
            print(f"Warning: Could not instantiate {tool_name}: {e}")
            # Try to use SerperDevTool as fallback
            if 'SerperDevTool' in available_tools and not any(type(t).__name__ == 'SerperDevTool' for t in tools):
                try:
                    tools.append(available_tools['SerperDevTool']())
                except Exception:
                    pass
    
    # Ensure we have at least one tool
    if not tools and 'SerperDevTool' in available_tools:
        try:
            tools.append(available_tools['SerperDevTool']())
        except Exception:
            print("Warning: Could not create fallback tool")
    
    return tools


# Note: Using actual CrewAI tools instead of custom implementations
# Tools are imported and instantiated directly from crewai_tools package
