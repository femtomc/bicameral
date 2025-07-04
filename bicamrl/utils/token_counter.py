"""Token counting utilities for better estimation and tracking."""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Token estimation ratios for different models
# Based on empirical data - characters per token
MODEL_CHAR_PER_TOKEN = {
    "claude": 3.5,  # Claude models average ~3.5 chars per token
    "gpt": 4.0,  # GPT models average ~4 chars per token
    "default": 4.0,  # Default fallback
}


class TokenCounter:
    """Provides better token estimation and tracking."""

    def __init__(self):
        self.session_tokens = 0
        self.current_message_tokens = 0

    def estimate_tokens(self, text: str, model: str = "default") -> int:
        """Estimate token count for text based on model type.

        This is still an approximation but better than word splitting.
        For accurate counts, we need the actual tokenizer.
        """
        if not text:
            return 0

        # Get the appropriate ratio
        if "claude" in model.lower():
            ratio = MODEL_CHAR_PER_TOKEN["claude"]
        elif "gpt" in model.lower():
            ratio = MODEL_CHAR_PER_TOKEN["gpt"]
        else:
            ratio = MODEL_CHAR_PER_TOKEN["default"]

        # Estimate based on character count
        # Add 10% buffer for special tokens
        estimated = int(len(text) / ratio * 1.1)

        return max(1, estimated)  # At least 1 token

    def update_session_tokens(self, tokens: int):
        """Update the session token count."""
        self.session_tokens += tokens

    def update_current_tokens(self, tokens: int):
        """Update current message token count (for display during streaming)."""
        self.current_message_tokens = tokens

    def get_session_tokens(self) -> int:
        """Get total tokens used in this session."""
        return self.session_tokens

    def reset_session(self):
        """Reset session token counter."""
        self.session_tokens = 0
        self.current_message_tokens = 0

    def parse_usage_info(self, usage: Dict) -> Tuple[int, int, int]:
        """Parse usage info from LLM response.

        Returns: (input_tokens, output_tokens, total_tokens)
        """
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # Some providers use different keys
        if input_tokens == 0:
            input_tokens = usage.get("prompt_tokens", 0)
        if output_tokens == 0:
            output_tokens = usage.get("completion_tokens", 0)

        total_tokens = input_tokens + output_tokens

        # If total_tokens is provided and doesn't match, use it
        if "total_tokens" in usage:
            provided_total = usage.get("total_tokens", 0)
            if provided_total != total_tokens and provided_total > 0:
                logger.warning(
                    f"Token count mismatch: calculated {total_tokens}, provided {provided_total}"
                )
                total_tokens = provided_total

        return input_tokens, output_tokens, total_tokens
