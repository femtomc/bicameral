"""Import conversation logs from Claude Code into bicamrl memory."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4

from ..core.interaction_model import (
    Interaction, Action, FeedbackType, ActionStatus
)
from ..core.memory import Memory
from ..utils.logging_config import get_logger

logger = get_logger("claude_code_importer")


class ClaudeCodeImporter:
    """Import conversation logs from Claude Code into bicamrl memory."""
    
    def __init__(self, memory: Memory):
        self.memory = memory
        self.interaction_cache = {}  # Track interactions being built
        
    async def import_directory(self, claude_dir: Path = None, project_filter: Optional[str] = None) -> Dict[str, int]:
        """Import all conversation logs from Claude Code directory.
        
        Args:
            claude_dir: Path to .claude directory (defaults to ~/.claude)
            project_filter: Optional project path filter (e.g., "/Users/femtomc/Dev/agents")
                           Only imports conversations from projects containing this path
        """
        if claude_dir is None:
            claude_dir = Path.home() / ".claude"
            
        projects_dir = claude_dir / "projects"
        if not projects_dir.exists():
            logger.error(f"Claude projects directory not found: {projects_dir}")
            return {"error": "Directory not found"}
            
        stats = {
            "sessions": 0,
            "interactions": 0,
            "actions": 0,
            "patterns": 0,
            "errors": 0,
            "skipped_projects": 0
        }
        
        # Process each project directory
        for project_dir in projects_dir.iterdir():
            if project_dir.is_dir():
                # Decode the project path from the directory name
                # Claude encodes paths by replacing "/" with "-"
                # Handle edge cases like paths with legitimate hyphens
                encoded_name = project_dir.name
                
                # Claude's encoding: absolute paths start with "-" (representing "/")
                if encoded_name.startswith("-"):
                    # This is an absolute path
                    parts = encoded_name.split("-")[1:]  # Skip the first empty part
                    decoded_path = "/" + "/".join(parts)
                else:
                    # Relative path (rare but handle it)
                    parts = encoded_name.split("-")
                    decoded_path = "/".join(parts)
                
                # Log the decoded path for debugging
                logger.debug(f"Decoded project path: {encoded_name} -> {decoded_path}")
                
                # Apply project filter if specified
                if project_filter:
                    if project_filter not in decoded_path:
                        logger.debug(f"Skipping project (filter mismatch): {decoded_path}")
                        stats["skipped_projects"] += 1
                        continue
                
                logger.info(f"Processing project: {decoded_path}")
                
                # Process each JSONL file in the project
                for jsonl_file in project_dir.glob("*.jsonl"):
                    try:
                        session_stats = await self.import_conversation_log(jsonl_file)
                        stats["sessions"] += 1
                        stats["interactions"] += session_stats.get("interactions", 0)
                        stats["actions"] += session_stats.get("actions", 0)
                        stats["patterns"] += session_stats.get("patterns", 0)
                    except Exception as e:
                        logger.error(f"Error importing {jsonl_file}: {e}")
                        stats["errors"] += 1
                        
        return stats
        
    async def import_conversation_log(self, log_path: Path) -> Dict[str, int]:
        """Import a single Claude Code conversation log file."""
        logger.info(f"Importing conversation log: {log_path}")
        
        stats = {
            "interactions": 0,
            "actions": 0,
            "patterns": 0
        }
        
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
        # Extract session ID from the first line
        if lines:
            first_event = json.loads(lines[0])
            session_id = first_event.get('sessionId', str(uuid4()))
        else:
            return stats
            
        current_interaction = None
        
        for line in lines:
            try:
                event = json.loads(line.strip())
                event_type = event.get('type')
                
                if event_type == 'user':
                    # Start new interaction
                    if current_interaction:
                        # Save previous interaction
                        await self._save_interaction(current_interaction)
                        stats["interactions"] += 1
                        stats["actions"] += len(current_interaction.actions_taken)
                        
                    current_interaction = self._create_interaction_from_user_event(event, session_id)
                    
                elif event_type == 'assistant' and current_interaction:
                    # Process assistant response
                    message = event.get('message', {})
                    content = message.get('content', [])
                    
                    for item in content:
                        if item.get('type') == 'tool_use':
                            action = self._parse_tool_use(item)
                            if action:
                                current_interaction.actions_taken.append(action)
                                
                elif event_type == 'user' and 'toolUseResult' in event and current_interaction:
                    # Update action result
                    tool_result = event.get('toolUseResult', {})
                    
                    # Extract tool_use_id from message content
                    message_content = event.get('message', {}).get('content', [])
                    tool_use_id = None
                    if isinstance(message_content, list) and message_content:
                        first_item = message_content[0]
                        if isinstance(first_item, dict):
                            tool_use_id = first_item.get('tool_use_id')
                    
                    # Find matching action and update result
                    if tool_use_id:
                        for action in current_interaction.actions_taken:
                            if action.details.get('tool_use_id') == tool_use_id:
                                self._update_action_result(action, tool_result)
                                break
                            
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                continue
            except Exception as e:
                logger.debug(f"Error parsing event type '{event_type}': {str(e)}")
                # Only log actual errors, not expected format variations
                if "expected str instance" not in str(e):
                    logger.error(f"Unexpected error parsing line: {e}", exc_info=True)
                continue
                
        # Save final interaction
        if current_interaction:
            await self._save_interaction(current_interaction)
            stats["interactions"] += 1
            stats["actions"] += len(current_interaction.actions_taken)
            
        # Skip pattern detection for now - can be run separately after import
        stats["patterns"] = 0
        
        return stats
        
    def _create_interaction_from_user_event(self, event: Dict, session_id: str) -> Interaction:
        """Create an Interaction object from a user event."""
        message = event.get('message', {})
        content = message.get('content', '')
        
        # Handle different content formats
        if isinstance(content, list):
            # Extract text from list of content items
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(str(item.get('text', '')))
                    elif 'content' in item:
                        text_parts.append(str(item.get('content', '')))
                elif isinstance(item, str):
                    text_parts.append(item)
                elif item is not None:
                    # Convert any other type to string
                    text_parts.append(str(item))
                # Skip None items
            # Ensure all parts are strings before joining
            content = ' '.join(str(part) for part in text_parts if part) if text_parts else ''
        elif not isinstance(content, str):
            content = str(content)
            
        # Parse timestamp
        timestamp_str = event.get('timestamp', '')
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()
            
        return Interaction(
            interaction_id=event.get('uuid', str(uuid4())),
            session_id=session_id,
            user_query=content,
            timestamp=timestamp,
            query_context={
                'cwd': event.get('cwd', ''),
                'version': event.get('version', '')
            },
            actions_taken=[],
            success=True  # Will update based on results
        )
        
    def _parse_tool_use(self, tool_use: Dict) -> Optional[Action]:
        """Convert Claude Code tool use to bicamrl Action."""
        tool_name = tool_use.get('name', '')
        tool_input = tool_use.get('input', {})
        tool_use_id = tool_use.get('id', '')
        
        # Map Claude Code tools to bicamrl action types
        action_mapping = {
            'str_replace_editor': 'edit_file',
            'str_replace_based_edit_tool': 'edit_file',
            'Read': 'read_file',
            'Write': 'write_file',
            'Edit': 'edit_file',
            'LS': 'list_directory',
            'Bash': 'run_command',
            'TodoWrite': 'manage_todos',
            'TodoRead': 'read_todos',
            'WebSearch': 'search_web',
            'WebFetch': 'fetch_url'
        }
        
        action_type = action_mapping.get(tool_name, tool_name.lower())
        
        # Extract target based on tool type
        target = None
        if tool_name in ['Read', 'Write', 'Edit', 'str_replace_editor', 'str_replace_based_edit_tool']:
            target = tool_input.get('file_path') or tool_input.get('path')
        elif tool_name == 'LS':
            target = tool_input.get('path', '.')
        elif tool_name == 'Bash':
            target = tool_input.get('command')
        elif tool_name in ['WebSearch', 'WebFetch']:
            target = tool_input.get('query') or tool_input.get('url')
            
        return Action(
            action_type=action_type,
            target=target,
            details={
                'tool_name': tool_name,
                'tool_input': tool_input,
                'tool_use_id': tool_use_id
            },
            status=ActionStatus.PLANNED
        )
        
    def _update_action_result(self, action: Action, tool_result: Dict):
        """Update action with tool execution result."""
        if isinstance(tool_result, dict):
            if tool_result.get('stdout'):
                action.result = tool_result['stdout']
                action.status = ActionStatus.COMPLETED
            elif tool_result.get('stderr'):
                action.error = tool_result['stderr']
                action.status = ActionStatus.FAILED
            else:
                # Other tool results
                action.result = str(tool_result)
                action.status = ActionStatus.COMPLETED
        elif isinstance(tool_result, str):
            if tool_result.startswith('Error:'):
                action.error = tool_result
                action.status = ActionStatus.FAILED
            else:
                action.result = tool_result
                action.status = ActionStatus.COMPLETED
                
    async def _save_interaction(self, interaction: Interaction):
        """Save interaction to memory."""
        try:
            # Complete the interaction
            interaction.execution_completed_at = datetime.now()
            
            # Determine success based on actions
            failed_actions = [a for a in interaction.actions_taken if a.status == ActionStatus.FAILED]
            interaction.success = len(failed_actions) == 0
            
            # Store in memory
            await self.memory.store_interaction(interaction)
            
        except Exception as e:
            logger.error(f"Error saving interaction: {e}")
            
    async def _detect_patterns_for_session(self, session_id: str) -> List[Dict]:
        """Run pattern detection on all interactions in a session."""
        try:
            # Get all interactions for the session
            # Use get_complete_interactions which allows filtering by session_id
            interactions = await self.memory.store.get_complete_interactions(
                session_id=session_id,
                limit=100
            )
            
            # Extract action sequences
            from ..core.pattern_detector import PatternDetector
            detector = PatternDetector(self.memory.store)
            
            patterns = []
            for i in range(len(interactions) - 2):  # Need at least 3 interactions
                action_sequence = []
                for j in range(i, min(i + 5, len(interactions))):  # Look at up to 5 interactions
                    for action in interactions[j].get('actions_taken', []):
                        action_sequence.append(action.get('action_type', ''))
                        
                if len(action_sequence) >= 3:
                    detected = await detector.detect_pattern(action_sequence)
                    patterns.extend(detected)
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []