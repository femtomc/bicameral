"""Process and apply developer feedback."""

from datetime import datetime
from typing import Any, Dict, Optional

from ..utils.logging_config import get_logger

logger = get_logger("feedback_processor")


class FeedbackProcessor:
    """Processes developer feedback to improve AI understanding."""

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    async def process_feedback(self, feedback_type: str, message: str) -> str:
        """Process different types of feedback."""
        # Get recent context for the feedback
        context = await self.memory_manager.get_recent_context()

        # Store the feedback
        feedback_data = {
            "type": feedback_type,
            "message": message,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }

        await self.memory_manager.store.add_feedback(feedback_data)

        # Process based on type
        if feedback_type == "correct":
            return await self._process_correction(message, context)
        elif feedback_type == "prefer":
            return await self._process_preference(message, context)
        elif feedback_type == "pattern":
            return await self._process_pattern(message, context)
        else:
            return f"Unknown feedback type: {feedback_type}"

    async def _process_correction(self, message: str, context: Dict) -> str:
        """Process a correction."""
        # Extract what was wrong and what's right
        # For now, we'll store it as negative feedback on recent patterns

        # Look for patterns that might have led to the error
        recent_actions = context.get("recent_actions", [])
        if recent_actions:
            # Mark recent patterns as less reliable
            patterns = await self.memory_manager.get_all_patterns()
            for pattern in patterns:
                # Simple heuristic: if pattern was recently used, reduce confidence
                pattern_actions = pattern.get("sequence", [])
                if pattern_actions and any(
                    a["action"] in pattern_actions for a in recent_actions[-5:]
                ):
                    await self.memory_manager.store.update_pattern_confidence(
                        pattern.get("id"),
                        pattern.get("confidence", 0.5) * 0.9,  # Reduce by 10%
                    )

        # Store the correction as a preference
        await self._extract_and_store_preference(message, "correction")

        return f"Correction noted: {message[:100]}..."

    async def _process_preference(self, message: str, context: Dict) -> str:
        """Process a preference."""
        # Try to extract key-value from message
        preference_data = self._parse_preference(message)

        if preference_data:
            await self.memory_manager.store.add_preference(
                {
                    "key": preference_data["key"],
                    "value": preference_data["value"],
                    "category": preference_data.get("category", "general"),
                    "source": "explicit",
                    "confidence": 1.0,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return f"Preference stored: {preference_data['key']} = {preference_data['value']}"
        else:
            # Store as general preference
            await self._extract_and_store_preference(message, "preference")
            return f"Preference noted: {message[:100]}..."

    async def _process_pattern(self, message: str, context: Dict) -> str:
        """Process a pattern teaching."""
        # Create a new pattern from the message
        pattern_data = {
            "name": f"User-taught: {message[:50]}...",
            "description": message,
            "pattern_type": "user_defined",
            "sequence": [],  # Will be filled by analyzing context
            "frequency": 1,
            "confidence": 0.8,  # High confidence for explicit teaching
            "source": "user_feedback",
            "timestamp": datetime.now().isoformat(),
        }

        # Try to extract sequence from recent actions
        recent_actions = context.get("recent_actions", [])
        if recent_actions:
            # Use last 5 actions as the pattern sequence
            pattern_data["sequence"] = [a["action"] for a in recent_actions[-5:]]

        await self.memory_manager.store.add_pattern(pattern_data)

        return f"Pattern learned: {message[:100]}..."

    def _parse_preference(self, message: str) -> Optional[Dict[str, Any]]:
        """Try to parse a preference from the message."""
        # Look for common patterns
        patterns = [
            # "use X instead of Y"
            (
                r"use\s+(.+?)\s+instead\s+of\s+(.+)",
                lambda m: {
                    "key": "prefer_over",
                    "value": {"prefer": m.group(1).strip(), "avoid": m.group(2).strip()},
                    "category": "style",
                },
            ),
            # "always X"
            (
                r"always\s+(.+)",
                lambda m: {"key": "always_do", "value": m.group(1).strip(), "category": "rules"},
            ),
            # "never X"
            (
                r"never\s+(.+)",
                lambda m: {"key": "never_do", "value": m.group(1).strip(), "category": "rules"},
            ),
            # "prefer X"
            (
                r"prefer\s+(.+)",
                lambda m: {"key": "preference", "value": m.group(1).strip(), "category": "style"},
            ),
            # "X: Y" format
            (
                r"^([^:]+):\s*(.+)$",
                lambda m: {
                    "key": m.group(1).strip(),
                    "value": m.group(2).strip(),
                    "category": "general",
                },
            ),
        ]

        import re

        for pattern, extractor in patterns:
            match = re.match(pattern, message, re.IGNORECASE)
            if match:
                return extractor(match)

        return None

    async def _extract_and_store_preference(self, message: str, pref_type: str) -> None:
        """Extract and store preference from free-form text."""
        # Simple extraction - store the whole message as a preference
        key = f"{pref_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        await self.memory_manager.store.add_preference(
            {
                "key": key,
                "value": message,
                "category": pref_type,
                "source": "feedback",
                "confidence": 0.9,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def apply_feedback_to_patterns(self) -> Dict[str, Any]:
        """Apply accumulated feedback to improve patterns."""
        feedback_items = await self.memory_manager.store.get_feedback()
        patterns = await self.memory_manager.get_all_patterns()

        applied_count = 0

        for feedback in feedback_items:
            if feedback.get("applied"):
                continue

            # Apply feedback based on type
            if feedback["type"] == "correct":
                # Reduce confidence in patterns that might have caused errors
                context = feedback.get("context", {})
                recent_actions = context.get("recent_actions", [])

                for pattern in patterns:
                    if self._pattern_matches_actions(pattern, recent_actions):
                        await self.memory_manager.store.update_pattern_confidence(
                            pattern.get("id"), pattern.get("confidence", 0.5) * 0.95
                        )
                        applied_count += 1

            # Mark feedback as applied
            await self.memory_manager.store.mark_feedback_applied(feedback.get("id"))

        return {"feedback_processed": len(feedback_items), "patterns_updated": applied_count}

    def _pattern_matches_actions(self, pattern: Dict, actions: list) -> bool:
        """Check if a pattern matches recent actions."""
        pattern_seq = pattern.get("sequence", [])
        if not pattern_seq:
            return False

        action_names = [a.get("action") for a in actions if a.get("action")]

        # Check if pattern sequence appears in recent actions
        for i in range(len(action_names) - len(pattern_seq) + 1):
            if action_names[i : i + len(pattern_seq)] == pattern_seq:
                return True

        return False
