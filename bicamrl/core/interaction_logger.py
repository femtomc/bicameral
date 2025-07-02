"""Logger for tracking complete user interactions."""

import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from .interaction_model import Interaction, Action, ActionStatus, FeedbackType
from ..utils.logging_config import get_logger

logger = get_logger("interaction_logger")


class InteractionLogger:
    """Tracks complete interaction cycles from query to feedback."""
    
    def __init__(self, memory):
        self.memory = memory
        self.current_interaction: Optional[Interaction] = None
        self.session_id = datetime.now().isoformat()
        
    def start_interaction(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Begin tracking a new interaction."""
        if self.current_interaction and not self.current_interaction.user_feedback:
            logger.warning("Starting new interaction without completing previous one")
            self.abandon_interaction()
        
        interaction_id = str(uuid.uuid4())
        self.current_interaction = Interaction(
            interaction_id=interaction_id,
            session_id=self.session_id,
            user_query=user_query,
            query_context=context or self._get_current_context()
        )
        
        logger.info(f"Started interaction {interaction_id}: {user_query[:50]}...")
        return interaction_id
    
    def log_interpretation(self, 
                         interpretation: str, 
                         planned_actions: List[str],
                         confidence: float = 0.0,
                         role: Optional[str] = None):
        """Record what the AI understood and plans to do."""
        if not self.current_interaction:
            raise ValueError("No active interaction to log interpretation")
        
        self.current_interaction.ai_interpretation = interpretation
        self.current_interaction.planned_actions = planned_actions
        self.current_interaction.confidence = confidence
        self.current_interaction.active_role = role
        self.current_interaction.execution_started_at = datetime.now()
        
        logger.info(f"Interpretation: {interpretation} (confidence: {confidence:.2f})")
        logger.info(f"Planned actions: {planned_actions}")
    
    def log_action(self, action_type: str, target: Optional[str] = None, 
                   details: Optional[Dict[str, Any]] = None) -> Action:
        """Log an individual action within the interaction."""
        if not self.current_interaction:
            raise ValueError("No active interaction to log action")
        
        action = Action(
            action_type=action_type,
            target=target,
            details=details or {},
            status=ActionStatus.EXECUTING,
            started_at=datetime.now()
        )
        
        self.current_interaction.actions_taken.append(action)
        logger.info(f"Executing action: {action_type} on {target}")
        
        return action
    
    def complete_action(self, action: Action, result: Optional[str] = None, 
                       error: Optional[str] = None):
        """Mark an action as completed."""
        action.completed_at = datetime.now()
        
        if error:
            action.status = ActionStatus.FAILED
            action.error = error
            logger.error(f"Action failed: {action.action_type} - {error}")
        else:
            action.status = ActionStatus.COMPLETED
            action.result = result
            logger.info(f"Action completed: {action.action_type}")
    
    def complete_interaction(self, 
                           feedback: Optional[str] = None,
                           feedback_type: Optional[FeedbackType] = None,
                           success: Optional[bool] = None,
                           tokens_used: int = 0):
        """Finalize and store the interaction."""
        if not self.current_interaction:
            logger.warning("No active interaction to complete")
            return None
        
        # Set completion time
        self.current_interaction.execution_completed_at = datetime.now()
        self.current_interaction.tokens_used = tokens_used
        
        # Process feedback
        if feedback:
            self.current_interaction.user_feedback = feedback
            
            # Auto-detect feedback type if not provided
            if not feedback_type:
                feedback_type = self._infer_feedback_type(feedback)
            self.current_interaction.feedback_type = feedback_type
            
            # Infer success if not explicitly provided
            if success is None:
                success = feedback_type in [FeedbackType.APPROVAL, FeedbackType.FOLLOWUP]
        
        self.current_interaction.success = success or False
        
        # Store the interaction
        interaction_dict = self.current_interaction.to_dict()
        asyncio.create_task(self._store_interaction(interaction_dict))
        
        logger.info(f"Completed interaction {self.current_interaction.interaction_id} "
                   f"with {len(self.current_interaction.actions_taken)} actions. "
                   f"Success: {self.current_interaction.success}")
        
        # Clear current interaction
        completed = self.current_interaction
        self.current_interaction = None
        
        return completed
    
    def abandon_interaction(self):
        """Abandon the current interaction without storing it."""
        if self.current_interaction:
            logger.warning(f"Abandoning interaction {self.current_interaction.interaction_id}")
            self.current_interaction = None
    
    async def _store_interaction(self, interaction_data: Dict[str, Any]):
        """Store interaction in the database."""
        await self.memory.store_interaction(self.current_interaction)
    
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current context (files, recent actions, etc.)."""
        # This will be populated by the actual MCP server
        return {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
    
    def _infer_feedback_type(self, feedback: str) -> FeedbackType:
        """Infer feedback type from the feedback text."""
        feedback_lower = feedback.lower()
        
        # Correction indicators
        correction_phrases = ['no,', 'not', 'actually', 'i meant', 'wrong', 'instead']
        if any(phrase in feedback_lower for phrase in correction_phrases):
            return FeedbackType.CORRECTION
        
        # Approval indicators
        approval_phrases = ['thanks', 'great', 'perfect', 'good', 'yes', 'correct', 'nice']
        if any(phrase in feedback_lower for phrase in approval_phrases):
            return FeedbackType.APPROVAL
        
        # Followup indicators
        followup_phrases = ['also', 'and', 'then', 'next', 'now', 'could you']
        if any(phrase in feedback_lower for phrase in followup_phrases):
            return FeedbackType.FOLLOWUP
        
        # Clarification indicators
        clarification_phrases = ['what i meant', 'to clarify', 'specifically', 'i mean']
        if any(phrase in feedback_lower for phrase in clarification_phrases):
            return FeedbackType.CLARIFICATION
        
        return FeedbackType.NONE


