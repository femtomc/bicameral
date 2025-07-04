"""Rate limiting decorator for MCP tools."""

import functools
from typing import Any, Callable, Dict

from ..utils.logging_config import get_logger

logger = get_logger("rate_limit_decorator")


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, retry_after: float, message: str = "Rate limit exceeded"):
        self.retry_after = retry_after
        self.message = message
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "error": "rate_limit_exceeded",
            "message": self.message,
            "retry_after": self.retry_after,
        }


def rate_limited(cost: float = 1.0):
    """
    Decorator to apply rate limiting to MCP tools.

    Args:
        cost: The cost of this operation (default 1.0)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to get the rate limiter from server context
            # This will be injected by the server during initialization
            rate_limiter = getattr(wrapper, "_rate_limiter", None)
            session_id = kwargs.get("_session_id", "default")

            if rate_limiter:
                # Check rate limit
                allowed, retry_after = await rate_limiter.check_rate_limit(session_id, cost)

                if not allowed:
                    logger.warning(
                        f"Rate limit exceeded for {func.__name__}",
                        extra={
                            "tool": func.__name__,
                            "session_id": session_id,
                            "retry_after": retry_after,
                        },
                    )
                    raise RateLimitExceeded(
                        retry_after=retry_after,
                        message=f"Rate limit exceeded. Please retry after {retry_after:.1f} seconds.",
                    )

            # Call the original function
            return await func(*args, **kwargs)

        # Store the cost for later reference
        wrapper._rate_limit_cost = cost
        return wrapper

    return decorator


def apply_rate_limiter_to_tools(mcp_instance, rate_limiter):
    """
    Apply rate limiter to all tools in an MCP instance.

    Args:
        mcp_instance: The FastMCP instance
        rate_limiter: The RateLimiter instance to use
    """
    # FastMCP uses _tools internally
    if hasattr(mcp_instance, "_tools"):
        tools = mcp_instance._tools
    elif hasattr(mcp_instance, "tools"):
        tools = mcp_instance.tools
    else:
        logger.warning("Could not find tools in MCP instance")
        return

    # Iterate through all tools and inject the rate limiter
    for tool_name, tool_handler in tools.items():
        if hasattr(tool_handler, "_rate_limit_cost"):
            # This tool has rate limiting enabled
            tool_handler._rate_limiter = rate_limiter
            logger.info(
                f"Rate limiter applied to tool: {tool_name}",
                extra={"tool": tool_name, "cost": tool_handler._rate_limit_cost},
            )
