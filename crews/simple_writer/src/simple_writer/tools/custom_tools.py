"""
Custom tools for simple_writer crew.

This module handles tool initialization and provides tools to agents.
"""

from typing import List, Any


def get_tools_for_agent(tool_names: List[str]) -> List[Any]:
    """Get actual CrewAI tools for an agent based on tool names."""
    # If no tools requested, return empty list
    if not tool_names or len(tool_names) == 0:
        return []
    
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
                print(f"Warning: Unknown tool '{tool_name}', skipping")
        except Exception as e:
            print(f"Warning: Could not instantiate {tool_name}: {e}")
    
    return tools


# Note: Using actual CrewAI tools instead of custom implementations
# Tools are imported and instantiated directly from crewai_tools package
