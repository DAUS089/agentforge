"""
Event bus patch to fix CrewAI EventBus errors.
"""

import warnings
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.task_events import TaskStartedEvent


def patch_event_bus():
    """Patch the event bus to handle None values gracefully."""
    
    # Store the original handler
    original_handlers = {}
    
    # Get all registered handlers for TaskStartedEvent
    if hasattr(crewai_event_bus, '_handlers') and TaskStartedEvent in crewai_event_bus._handlers:
        original_handlers[TaskStartedEvent] = crewai_event_bus._handlers[TaskStartedEvent].copy()
        
        # Clear the problematic handlers
        crewai_event_bus._handlers[TaskStartedEvent] = []
        
        # Add a safe handler
        @crewai_event_bus.on(TaskStartedEvent)
        def safe_on_task_started(source, event: TaskStartedEvent):
            try:
                # Check if source has required attributes
                if not hasattr(source, 'agent') or source.agent is None:
                    return
                if not hasattr(source.agent, 'crew') or source.agent.crew is None:
                    return
                    
                # Call original handlers if they exist
                if TaskStartedEvent in original_handlers:
                    for handler in original_handlers[TaskStartedEvent]:
                        try:
                            handler(source, event)
                        except Exception as e:
                            # Suppress the error but log it
                            warnings.warn(f"Event handler failed: {e}", UserWarning)
                            
            except Exception as e:
                # Suppress all errors from event handling
                warnings.warn(f"Event handling failed: {e}", UserWarning)


def apply_patch():
    """Apply the event bus patch and emoji encoding fix."""
    try:
        # Apply emoji encoding fix first
        patch_emoji_encoding()
        
        # Apply event bus patch
        patch_event_bus()
        print("[INFO] Event bus patch applied successfully")
    except Exception as e:
        print(f"[WARNING] Could not apply event bus patch: {e}")


def patch_emoji_encoding():
    """Patch stdout to handle emoji encoding issues."""
    import sys
    
    # Store original stdout
    original_stdout = sys.stdout
    
    class SafeStdout:
        def __init__(self, original_stdout):
            self.original_stdout = original_stdout
            
        def write(self, text):
            # Replace emojis with safe text
            safe_text = text.replace('🚀', '[START]').replace('⚡', '[LIGHTNING]').replace('🔥', '[FIRE]')
            safe_text = safe_text.replace('📊', '[CHART]').replace('💡', '[IDEA]').replace('🔍', '[SEARCH]')
            safe_text = safe_text.replace('✅', '[OK]').replace('❌', '[ERROR]').replace('⚠', '[WARN]')
            safe_text = safe_text.replace('🤖', '[AI]').replace('📋', '[INFO]').replace('🔧', '[TOOL]')
            safe_text = safe_text.replace('📁', '[FOLDER]').replace('🚨', '[ALERT]').replace('⚒', '[HAMMER]')
            safe_text = safe_text.replace('🔄', '[REFRESH]').replace('📄', '[DOCUMENT]').replace('📦', '[PACKAGE]')
            safe_text = safe_text.replace('🛠', '[TOOLS]').replace('💾', '[SAVE]').replace('🎭', '[MASK]')
            safe_text = safe_text.replace('👥', '[PEOPLE]').replace('🎨', '[ART]').replace('✨', '[SPARKLE]')
            safe_text = safe_text.replace('📚', '[BOOKS]').replace('🎉', '[PARTY]').replace('🏃', '[RUN]')
            safe_text = safe_text.replace('🧪', '[TEST]').replace('🎓', '[GRADUATE]').replace('🎯', '[TARGET]')
            safe_text = safe_text.replace('•', '[BULLET]').replace('═', '=').replace('║', '|')
            safe_text = safe_text.replace('█', '#').replace('╗', '+').replace('╚', '+')
            safe_text = safe_text.replace('╝', '+').replace('╔', '+').replace('└', '+')
            safe_text = safe_text.replace('├', '+').replace('─', '-').replace('️', '')
            
            try:
                self.original_stdout.write(safe_text)
            except UnicodeEncodeError:
                # If still having encoding issues, encode as ASCII with replacement
                safe_text = safe_text.encode('ascii', 'replace').decode('ascii')
                self.original_stdout.write(safe_text)
                
        def flush(self):
            self.original_stdout.flush()
            
        def __getattr__(self, name):
            return getattr(self.original_stdout, name)
    
    # Apply the safe stdout wrapper
    sys.stdout = SafeStdout(original_stdout)
