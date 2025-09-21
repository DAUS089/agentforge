"""
Crew Templates and Pattern Library for agentforge.

This module provides pre-built crew patterns for common use cases,
making it easier for users to get started with specific types of tasks.
"""

from .crew_template_library import CrewTemplateLibrary, CrewTemplate
from .template_manager import TemplateManager

__all__ = ["CrewTemplateLibrary", "CrewTemplate", "TemplateManager"]
