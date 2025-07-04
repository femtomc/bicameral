"""Rate limiting functionality for Bicamrl server."""

import asyncio
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

from .utils.logging_config import get_logger

logger = get_logger("rate_limiter")


class RateLimiter:
    """Token bucket rate limiter with sliding window."""

    def __init__(
        self, requests_per_minute: int = 60, burst_size: int = 10, window_seconds: int = 60
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst requests allowed
            window_seconds: Time window for rate limiting
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.window_seconds = window_seconds

        # Track requests per client (session_id or IP)
        self.request_windows: Dict[str, deque] = defaultdict(deque)

        # Token buckets for burst control
        self.token_buckets: Dict[str, float] = defaultdict(lambda: float(burst_size))
        self.last_refill: Dict[str, float] = defaultdict(time.time)

        # Lock for thread safety
        self.lock = asyncio.Lock()

        # Metrics
        self.total_requests = 0
        self.rejected_requests = 0
        self.client_metrics = defaultdict(lambda: {"allowed": 0, "rejected": 0})

    async def check_rate_limit(
        self, client_id: str, cost: float = 1.0
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if request is allowed under rate limit.

        Args:
            client_id: Unique identifier for the client
            cost: Cost of this request (default 1.0)

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        async with self.lock:
            now = time.time()

            # Clean old requests from sliding window
            self._clean_window(client_id, now)

            # Refill token bucket
            self._refill_tokens(client_id, now)

            # Check sliding window rate limit
            window_count = len(self.request_windows[client_id])
            if window_count >= self.requests_per_minute:
                # Calculate when the oldest request will expire
                oldest_request = self.request_windows[client_id][0]
                retry_after = (oldest_request + self.window_seconds) - now

                self.rejected_requests += 1
                self.client_metrics[client_id]["rejected"] += 1

                logger.warning(
                    f"Rate limit exceeded for client {client_id}",
                    extra={
                        "client_id": client_id,
                        "window_count": window_count,
                        "limit": self.requests_per_minute,
                        "retry_after": retry_after,
                    },
                )

                return False, retry_after

            # Check token bucket for burst control
            if self.token_buckets[client_id] < cost:
                # Not enough tokens for this request
                tokens_needed = cost - self.token_buckets[client_id]
                refill_rate = self.requests_per_minute / 60.0  # tokens per second
                retry_after = tokens_needed / refill_rate

                self.rejected_requests += 1
                self.client_metrics[client_id]["rejected"] += 1

                logger.warning(
                    f"Burst limit exceeded for client {client_id}",
                    extra={
                        "client_id": client_id,
                        "tokens_available": self.token_buckets[client_id],
                        "cost": cost,
                        "retry_after": retry_after,
                    },
                )

                return False, retry_after

            # Request is allowed
            self.request_windows[client_id].append(now)
            self.token_buckets[client_id] -= cost
            self.total_requests += 1
            self.client_metrics[client_id]["allowed"] += 1

            return True, None

    def _clean_window(self, client_id: str, now: float):
        """Remove requests outside the sliding window."""
        cutoff = now - self.window_seconds
        while self.request_windows[client_id] and self.request_windows[client_id][0] < cutoff:
            self.request_windows[client_id].popleft()

    def _refill_tokens(self, client_id: str, now: float):
        """Refill tokens based on elapsed time."""
        elapsed = now - self.last_refill[client_id]
        refill_rate = self.requests_per_minute / 60.0  # tokens per second
        tokens_to_add = elapsed * refill_rate

        self.token_buckets[client_id] = min(
            self.burst_size, self.token_buckets[client_id] + tokens_to_add
        )
        self.last_refill[client_id] = now

    def get_metrics(self) -> Dict[str, any]:
        """Get rate limiter metrics."""
        return {
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "rejection_rate": self.rejected_requests / max(1, self.total_requests),
            "active_clients": len(self.request_windows),
            "config": {
                "requests_per_minute": self.requests_per_minute,
                "burst_size": self.burst_size,
                "window_seconds": self.window_seconds,
            },
        }

    def get_client_status(self, client_id: str) -> Dict[str, any]:
        """Get rate limit status for a specific client."""
        now = time.time()
        self._clean_window(client_id, now)
        self._refill_tokens(client_id, now)

        return {
            "client_id": client_id,
            "requests_in_window": len(self.request_windows[client_id]),
            "tokens_available": self.token_buckets[client_id],
            "metrics": dict(self.client_metrics[client_id]),
        }

    async def reset_client(self, client_id: str):
        """Reset rate limit for a specific client."""
        async with self.lock:
            self.request_windows[client_id].clear()
            self.token_buckets[client_id] = float(self.burst_size)
            self.last_refill[client_id] = time.time()
            self.client_metrics[client_id] = {"allowed": 0, "rejected": 0}

            logger.info(f"Rate limit reset for client {client_id}")


class RateLimitMiddleware:
    """Middleware for applying rate limiting to MCP tools."""

    def __init__(self, rate_limiter: RateLimiter, cost_mapping: Optional[Dict[str, float]] = None):
        """
        Initialize middleware.

        Args:
            rate_limiter: The rate limiter instance
            cost_mapping: Optional mapping of tool names to their costs
        """
        self.rate_limiter = rate_limiter
        self.cost_mapping = cost_mapping or {}
        self.default_cost = 1.0

    def get_tool_cost(self, tool_name: str) -> float:
        """Get the cost for a specific tool."""
        return self.cost_mapping.get(tool_name, self.default_cost)

    async def check_tool_limit(
        self, tool_name: str, session_id: str
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if a tool call is allowed.

        Args:
            tool_name: Name of the tool being called
            session_id: Session identifier

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        cost = self.get_tool_cost(tool_name)
        return await self.rate_limiter.check_rate_limit(session_id, cost)


# Default rate limiter configurations
def create_default_rate_limiter() -> RateLimiter:
    """Create rate limiter with default production settings."""
    return RateLimiter(requests_per_minute=60, burst_size=10, window_seconds=60)


def create_strict_rate_limiter() -> RateLimiter:
    """Create strict rate limiter for resource-intensive operations."""
    return RateLimiter(requests_per_minute=20, burst_size=5, window_seconds=60)


def create_relaxed_rate_limiter() -> RateLimiter:
    """Create relaxed rate limiter for lightweight operations."""
    return RateLimiter(requests_per_minute=120, burst_size=20, window_seconds=60)
