"""
Template Manager for agentforge.

Handles template operations, customization, and integration with crew generation.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import yaml
from .crew_template_library import CrewTemplate, CrewTemplateLibrary, AgentTemplate, TaskTemplate


class TemplateManager:
    """Manages crew templates and their customization."""
    
    def __init__(self):
        self.library = CrewTemplateLibrary()
        self.custom_templates: Dict[str, CrewTemplate] = {}
        self.template_cache: Dict[str, Dict[str, Any]] = {}
    
    def get_template(self, template_name: str) -> Optional[CrewTemplate]:
        """Get a template by name (built-in or custom)."""
        # Check custom templates first
        if template_name in self.custom_templates:
            return self.custom_templates[template_name]
        
        # Then check built-in templates
        return self.library.get_template(template_name)
    
    def list_templates(self, include_custom: bool = True) -> List[str]:
        """List all available templates."""
        templates = self.library.list_templates()
        if include_custom:
            templates.extend(self.custom_templates.keys())
        return templates
    
    def search_templates(self, query: str) -> List[CrewTemplate]:
        """Search templates by query."""
        results = self.library.search_templates(query)
        
        # Also search custom templates
        query_lower = query.lower()
        for template in self.custom_templates.values():
            if (query_lower in template.name.lower() or 
                query_lower in template.description.lower() or
                any(query_lower in use_case.lower() for use_case in template.use_cases)):
                results.append(template)
        
        return results
    
    def create_custom_template(self, template: CrewTemplate) -> bool:
        """Create a custom template."""
        try:
            self.custom_templates[template.name.lower()] = template
            return True
        except Exception:
            return False
    
    def update_template(self, template_name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing template."""
        template = self.get_template(template_name)
        if not template:
            return False
        
        try:
            # Update template fields
            for key, value in updates.items():
                if hasattr(template, key):
                    setattr(template, key, value)
            
            # Save to custom templates if it's a built-in template
            if template_name in self.library.list_templates():
                self.custom_templates[template_name.lower()] = template
            
            return True
        except Exception:
            return False
    
    def delete_custom_template(self, template_name: str) -> bool:
        """Delete a custom template."""
        if template_name in self.custom_templates:
            del self.custom_templates[template_name]
            return True
        return False
    
    def export_template(self, template_name: str, file_path: str) -> bool:
        """Export a template to a file."""
        template = self.get_template(template_name)
        if not template:
            return False
        
        try:
            template_data = self._template_to_dict(template)
            
            if file_path.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(template_data, f, indent=2)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                with open(file_path, 'w') as f:
                    yaml.dump(template_data, f, default_flow_style=False)
            else:
                return False
            
            return True
        except Exception:
            return False
    
    def import_template(self, file_path: str) -> bool:
        """Import a template from a file."""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    template_data = json.load(f)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                with open(file_path, 'r') as f:
                    template_data = yaml.safe_load(f)
            else:
                return False
            
            template = self._dict_to_template(template_data)
            if template:
                self.custom_templates[template.name.lower()] = template
                return True
            
            return False
        except Exception:
            return False
    
    def customize_template(self, template_name: str, customizations: Dict[str, Any]) -> Optional[CrewTemplate]:
        """Create a customized version of a template."""
        base_template = self.get_template(template_name)
        if not base_template:
            return None
        
        try:
            # Create a copy of the template
            template_data = self._template_to_dict(base_template)
            
            # Apply customizations
            self._apply_customizations(template_data, customizations)
            
            # Create new template
            customized_template = self._dict_to_template(template_data)
            if customized_template:
                # Add customization suffix to name
                customized_template.name = f"{base_template.name} (Customized)"
                return customized_template
            
            return None
        except Exception:
            return None
    
    def get_template_recommendations(self, task_description: str) -> List[CrewTemplate]:
        """Get template recommendations based on task description."""
        recommendations = []
        task_lower = task_description.lower()
        
        # Keywords for different template types
        keyword_mapping = {
            "data_analysis": ["data", "analysis", "analytics", "statistics", "metrics"],
            "web_scraping": ["scrape", "crawl", "extract", "web", "website"],
            "content_creation": ["content", "write", "article", "blog", "documentation"],
            "code_review": ["code", "review", "bug", "debug", "programming"],
            "research_crew": ["research", "study", "investigate", "academic"],
            "customer_support": ["support", "help", "ticket", "customer", "issue"],
            "marketing_automation": ["marketing", "campaign", "promotion", "advertising"],
            "financial_analysis": ["financial", "finance", "budget", "investment", "money"],
            "bug_triage": ["bug", "triage", "issue", "defect", "error"],
            "documentation": ["documentation", "docs", "manual", "guide", "tutorial"]
        }
        
        # Score templates based on keyword matches
        template_scores = {}
        for template_name, keywords in keyword_mapping.items():
            score = sum(1 for keyword in keywords if keyword in task_lower)
            if score > 0:
                template = self.library.get_template(template_name)
                if template:
                    template_scores[template] = score
        
        # Sort by score and return top recommendations
        sorted_templates = sorted(template_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [template for template, score in sorted_templates[:5]]
        
        return recommendations
    
    def _template_to_dict(self, template: CrewTemplate) -> Dict[str, Any]:
        """Convert a template to dictionary format."""
        return {
            "name": template.name,
            "description": template.description,
            "category": template.category,
            "workflow": template.workflow.value,
            "agents": [
                {
                    "name": agent.name,
                    "role": agent.role,
                    "goal": agent.goal,
                    "backstory": agent.backstory,
                    "tools": agent.tools,
                    "memory_type": agent.memory_type,
                    "max_iter": agent.max_iter,
                    "verbose": agent.verbose
                }
                for agent in template.agents
            ],
            "tasks": [
                {
                    "name": task.name,
                    "description": task.description,
                    "expected_output": task.expected_output,
                    "context": task.context,
                    "tools": task.tools
                }
                for task in template.tasks
            ],
            "tools": template.tools,
            "estimated_duration": template.estimated_duration,
            "complexity": template.complexity,
            "use_cases": template.use_cases
        }
    
    def _dict_to_template(self, template_data: Dict[str, Any]) -> Optional[CrewTemplate]:
        """Convert dictionary to template object."""
        try:
            from .crew_template_library import WorkflowType
            
            agents = [
                AgentTemplate(
                    name=agent_data["name"],
                    role=agent_data["role"],
                    goal=agent_data["goal"],
                    backstory=agent_data["backstory"],
                    tools=agent_data["tools"],
                    memory_type=agent_data.get("memory_type", "conversation_buffer"),
                    max_iter=agent_data.get("max_iter", 5),
                    verbose=agent_data.get("verbose", True)
                )
                for agent_data in template_data["agents"]
            ]
            
            tasks = [
                TaskTemplate(
                    name=task_data["name"],
                    description=task_data["description"],
                    expected_output=task_data["expected_output"],
                    context=task_data.get("context"),
                    tools=task_data.get("tools")
                )
                for task_data in template_data["tasks"]
            ]
            
            workflow = WorkflowType(template_data["workflow"])
            
            return CrewTemplate(
                name=template_data["name"],
                description=template_data["description"],
                category=template_data["category"],
                workflow=workflow,
                agents=agents,
                tasks=tasks,
                tools=template_data["tools"],
                estimated_duration=template_data["estimated_duration"],
                complexity=template_data["complexity"],
                use_cases=template_data["use_cases"]
            )
        except Exception:
            return None
    
    def _apply_customizations(self, template_data: Dict[str, Any], customizations: Dict[str, Any]):
        """Apply customizations to template data."""
        for key, value in customizations.items():
            if key in template_data:
                if isinstance(value, dict) and isinstance(template_data[key], dict):
                    template_data[key].update(value)
                else:
                    template_data[key] = value
            elif key == "add_agent" and isinstance(value, dict):
                template_data["agents"].append(value)
            elif key == "add_task" and isinstance(value, dict):
                template_data["tasks"].append(value)
            elif key == "add_tool" and isinstance(value, str):
                if value not in template_data["tools"]:
                    template_data["tools"].append(value)
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """Get statistics about available templates."""
        built_in_count = len(self.library.list_templates())
        custom_count = len(self.custom_templates)
        
        # Count by category
        category_counts = {}
        for template in self.library._templates.values():
            category = template.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for template in self.custom_templates.values():
            category = template.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_templates": built_in_count + custom_count,
            "built_in_templates": built_in_count,
            "custom_templates": custom_count,
            "categories": category_counts
        }
