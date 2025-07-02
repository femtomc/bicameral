"""Command role definitions and structures for behavioral templates."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TriggerType(str, Enum):
    """Types of context triggers."""
    FILE_PATTERN = "file_pattern"
    TASK_KEYWORD = "task_keyword"
    INTERACTION_PATTERN = "interaction_pattern"
    TIME_OF_DAY = "time_of_day"
    ERROR_STATE = "error_state"
    USER_REQUEST = "user_request"


class CommunicationStyle(str, Enum):
    """Communication style preferences."""
    CONCISE = "concise"  # Minimal explanations, focus on action
    EXPLANATORY = "explanatory"  # Detailed explanations of actions
    INTERACTIVE = "interactive"  # Ask questions, confirm before acting
    AUTONOMOUS = "autonomous"  # Act independently, report results


@dataclass
class ContextTrigger:
    """A condition that may activate a role."""
    trigger_type: TriggerType
    pattern: str  # Regex or keyword pattern
    weight: float = 1.0  # Importance of this trigger
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, context: Dict[str, Any]) -> float:
        """Check if trigger matches context, return confidence score."""
        if self.trigger_type == TriggerType.FILE_PATTERN:
            files = context.get("files", [])
            import re
            pattern = re.compile(self.pattern)
            matches = sum(1 for f in files if pattern.search(f))
            return min(1.0, matches / max(1, len(files))) * self.weight
            
        elif self.trigger_type == TriggerType.TASK_KEYWORD:
            task = context.get("task_description", "").lower()
            keywords = self.pattern.lower().split(",")
            matches = sum(1 for kw in keywords if kw.strip() in task)
            return min(1.0, matches / len(keywords)) * self.weight
            
        elif self.trigger_type == TriggerType.INTERACTION_PATTERN:
            recent_actions = context.get("recent_actions", [])
            pattern_actions = self.pattern.split("->")
            if len(recent_actions) >= len(pattern_actions):
                # Check if pattern matches recent actions
                match_score = sum(
                    1 for i, action in enumerate(pattern_actions)
                    if i < len(recent_actions) and action in recent_actions[-(i+1)]
                ) / len(pattern_actions)
                return match_score * self.weight
                
        return 0.0


@dataclass
class DecisionRule:
    """A rule that guides decision-making in a role."""
    condition: str  # Natural language condition
    action: str  # What to do when condition is met
    priority: int = 0  # Higher priority rules evaluated first
    
    def to_prompt_instruction(self) -> str:
        """Convert to prompt instruction."""
        return f"When {self.condition}, {self.action}"


@dataclass
class CommunicationProfile:
    """Profile for how to communicate in this role."""
    style: CommunicationStyle
    verbosity: float = 0.5  # 0.0 = very terse, 1.0 = very verbose
    proactivity: float = 0.5  # 0.0 = only respond, 1.0 = very proactive
    question_frequency: float = 0.3  # How often to ask clarifying questions
    
    def to_prompt_modifiers(self) -> List[str]:
        """Convert to prompt modifiers."""
        modifiers = []
        
        if self.style == CommunicationStyle.CONCISE:
            modifiers.append("Be concise and focus on actions rather than explanations.")
        elif self.style == CommunicationStyle.EXPLANATORY:
            modifiers.append("Explain your reasoning and actions clearly.")
        elif self.style == CommunicationStyle.INTERACTIVE:
            modifiers.append("Ask clarifying questions when uncertain.")
        elif self.style == CommunicationStyle.AUTONOMOUS:
            modifiers.append("Work independently and report results.")
            
        if self.verbosity < 0.3:
            modifiers.append("Use minimal words.")
        elif self.verbosity > 0.7:
            modifiers.append("Provide detailed information.")
            
        if self.proactivity > 0.7:
            modifiers.append("Anticipate needs and suggest next steps.")
            
        return modifiers


class CommandRole(BaseModel):
    """A behavioral template for the wake layer."""
    
    # Identity
    name: str = Field(..., description="Role name")
    description: str = Field(..., description="Role description")
    
    # Activation
    context_triggers: List[ContextTrigger] = Field(
        default_factory=list,
        description="Conditions that activate this role"
    )
    confidence_threshold: float = Field(
        0.7,
        description="Minimum confidence to activate role"
    )
    
    # Behavior
    tool_preferences: Dict[str, float] = Field(
        default_factory=dict,
        description="Tool name -> preference weight"
    )
    decision_rules: List[DecisionRule] = Field(
        default_factory=list,
        description="Rules guiding behavior"
    )
    communication_profile: CommunicationProfile = Field(
        default_factory=lambda: CommunicationProfile(CommunicationStyle.EXPLANATORY),
        description="Communication preferences"
    )
    
    # Learning
    successful_patterns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Patterns that worked well"
    )
    failure_patterns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Patterns to avoid"
    )
    
    # Metrics
    success_rate: float = Field(0.0, description="Historical success rate")
    usage_count: int = Field(0, description="Times role has been used")
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Last update time"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation time"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    def calculate_activation_score(self, context: Dict[str, Any]) -> float:
        """Calculate how well this role matches the current context."""
        if not self.context_triggers:
            return 0.0
            
        scores = [trigger.matches(context) for trigger in self.context_triggers]
        # Use weighted average of trigger scores
        return sum(scores) / len(scores)
    
    def should_activate(self, context: Dict[str, Any]) -> bool:
        """Check if role should activate given context."""
        score = self.calculate_activation_score(context)
        return score >= self.confidence_threshold
    
    def to_prompt_context(self) -> str:
        """Convert role to prompt context instructions."""
        lines = [
            f"You are operating in '{self.name}' role: {self.description}",
            "",
            "Behavioral Guidelines:"
        ]
        
        # Add communication style
        for modifier in self.communication_profile.to_prompt_modifiers():
            lines.append(f"- {modifier}")
        
        # Add decision rules
        if self.decision_rules:
            lines.append("")
            lines.append("Decision Rules:")
            for rule in sorted(self.decision_rules, key=lambda r: -r.priority):
                lines.append(f"- {rule.to_prompt_instruction()}")
        
        # Add tool preferences
        if self.tool_preferences:
            lines.append("")
            lines.append("Tool Preferences:")
            sorted_tools = sorted(
                self.tool_preferences.items(),
                key=lambda x: -x[1]
            )
            for tool, weight in sorted_tools[:5]:  # Top 5 tools
                if weight > 0.5:
                    lines.append(f"- Prefer using '{tool}' (weight: {weight:.1f})")
        
        return "\n".join(lines)
    
    def update_metrics(self, success: bool):
        """Update role performance metrics."""
        self.usage_count += 1
        # Exponential moving average for success rate
        alpha = 0.1  # Learning rate
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        self.last_updated = datetime.now()
    
    def to_markdown(self) -> str:
        """Export role as a detailed Markdown document."""
        lines = [
            f"# {self.name}",
            "",
            f"_{self.description}_",
            "",
            "## Overview",
            "",
            f"- **Confidence Threshold**: {self.confidence_threshold:.2f}",
            f"- **Communication Style**: {self.communication_profile.style.value}",
            f"- **Success Rate**: {self.success_rate:.2%}",
            f"- **Usage Count**: {self.usage_count}",
            f"- **Last Updated**: {self.last_updated.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Activation Triggers",
            ""
        ]
        
        # Group triggers by type
        trigger_groups = {}
        for trigger in self.context_triggers:
            if trigger.trigger_type not in trigger_groups:
                trigger_groups[trigger.trigger_type] = []
            trigger_groups[trigger.trigger_type].append(trigger)
        
        for trigger_type, triggers in trigger_groups.items():
            lines.append(f"### {trigger_type.value.replace('_', ' ').title()}")
            lines.append("")
            for trigger in sorted(triggers, key=lambda t: -t.weight):
                lines.append(f"- **{trigger.pattern}** (weight: {trigger.weight:.2f})")
            lines.append("")
        
        # Decision rules
        if self.decision_rules:
            lines.extend([
                "## Decision Rules",
                ""
            ])
            for i, rule in enumerate(self.decision_rules):
                lines.append(f"{i+1}. **When**: {rule.condition}")
                lines.append(f"   **Then**: {rule.action}")
                if rule.priority > 0:
                    lines.append(f"   **Priority**: {rule.priority}")
                lines.append("")
        
        # Tool preferences
        if self.tool_preferences:
            lines.extend([
                "## Tool Preferences",
                ""
            ])
            sorted_tools = sorted(self.tool_preferences.items(), key=lambda x: -x[1])
            for tool, weight in sorted_tools:
                if weight > 0.3:  # Only show significant preferences
                    lines.append(f"- **{tool}**: {weight:.2f}")
            lines.append("")
        
        # Communication profile
        lines.extend([
            "## Communication Profile",
            "",
            f"- **Style**: {self.communication_profile.style.value}",
            f"- **Verbosity**: {self.communication_profile.verbosity:.2f}",
            f"- **Proactivity**: {self.communication_profile.proactivity:.2f}",
            f"- **Question Frequency**: {self.communication_profile.question_frequency:.2f}",
            ""
        ])
        
        # Add modifiers based on profile values
        lines.append("### Communication Modifiers")
        lines.append("")
        
        if self.communication_profile.verbosity < 0.3:
            lines.append("- Use minimal words")
        elif self.communication_profile.verbosity > 0.7:
            lines.append("- Provide detailed information")
            
        if self.communication_profile.proactivity > 0.7:
            lines.append("- Anticipate needs and suggest next steps")
        elif self.communication_profile.proactivity < 0.3:
            lines.append("- Wait for explicit requests before acting")
            
        if self.communication_profile.question_frequency > 0.7:
            lines.append("- Ask clarifying questions frequently")
        elif self.communication_profile.question_frequency < 0.3:
            lines.append("- Minimize questions, make reasonable assumptions")
            
        lines.append("")
        
        # Successful patterns (if any)
        if self.successful_patterns:
            lines.extend([
                "## Successful Patterns",
                "",
                "Patterns that have worked well with this role:",
                ""
            ])
            for pattern in self.successful_patterns[:5]:  # Limit to 5
                lines.append(f"- {pattern}")
            lines.append("")
        
        # System prompt section
        lines.extend([
            "## System Prompt",
            "",
            "Use this role by incorporating the following into your system prompt:",
            "",
            "```",
            self.to_prompt_context(),
            "```"
        ])
        
        return "\n".join(lines)


# No pre-defined roles - all roles will be discovered from actual usage patterns
BUILTIN_ROLES = []

# Example role structure for reference:
# CommandRole(
#     name="Discovered Role Name",
#     description="Role description based on observed patterns",
#     context_triggers=[...],
#     tool_preferences={...},
#     decision_rules=[...],
#     communication_profile=CommunicationProfile(...)
# )