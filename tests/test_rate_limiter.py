"""Tests for rate limiting functionality."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

from bicamrl.rate_limiter import RateLimiter, RateLimitMiddleware
from bicamrl.utils.rate_limit_decorator import rate_limited, RateLimitExceeded


@pytest.mark.asyncio
async def test_rate_limiter_allows_requests_under_limit():
    """Test that requests under the rate limit are allowed."""
    limiter = RateLimiter(requests_per_minute=60, burst_size=10)
    
    # Should allow 10 requests in burst
    for i in range(10):
        allowed, retry_after = await limiter.check_rate_limit("test_client")
        assert allowed is True
        assert retry_after is None
    
    # Check metrics
    metrics = limiter.get_metrics()
    assert metrics["total_requests"] == 10
    assert metrics["rejected_requests"] == 0


@pytest.mark.asyncio
async def test_rate_limiter_blocks_burst_exceeded():
    """Test that requests exceeding burst size are blocked."""
    limiter = RateLimiter(requests_per_minute=60, burst_size=5)
    
    # Should allow 5 requests
    for i in range(5):
        allowed, retry_after = await limiter.check_rate_limit("test_client")
        assert allowed is True
    
    # 6th request should be blocked
    allowed, retry_after = await limiter.check_rate_limit("test_client")
    assert allowed is False
    assert retry_after > 0
    
    # Check metrics
    metrics = limiter.get_metrics()
    assert metrics["total_requests"] == 5
    assert metrics["rejected_requests"] == 1


@pytest.mark.asyncio
async def test_rate_limiter_sliding_window():
    """Test sliding window rate limiting."""
    limiter = RateLimiter(requests_per_minute=10, burst_size=10, window_seconds=1)
    
    # Make 10 requests (fills the window)
    for i in range(10):
        allowed, retry_after = await limiter.check_rate_limit("test_client")
        assert allowed is True
    
    # 11th request should be blocked
    allowed, retry_after = await limiter.check_rate_limit("test_client")
    assert allowed is False
    
    # Wait for window to slide
    await asyncio.sleep(1.1)
    
    # Should allow new requests
    allowed, retry_after = await limiter.check_rate_limit("test_client")
    assert allowed is True


@pytest.mark.asyncio
async def test_rate_limiter_multiple_clients():
    """Test rate limiting with multiple clients."""
    limiter = RateLimiter(requests_per_minute=60, burst_size=5)
    
    # Client 1 makes 5 requests
    for i in range(5):
        allowed, _ = await limiter.check_rate_limit("client1")
        assert allowed is True
    
    # Client 1's 6th request is blocked
    allowed, _ = await limiter.check_rate_limit("client1")
    assert allowed is False
    
    # Client 2 can still make requests
    for i in range(5):
        allowed, _ = await limiter.check_rate_limit("client2")
        assert allowed is True


@pytest.mark.asyncio
async def test_rate_limiter_token_refill():
    """Test token bucket refill mechanism."""
    limiter = RateLimiter(requests_per_minute=60, burst_size=2)
    
    # Use up tokens
    await limiter.check_rate_limit("test_client", cost=2.0)
    
    # Next request should fail
    allowed, retry_after = await limiter.check_rate_limit("test_client")
    assert allowed is False
    
    # Wait for token refill (60 req/min = 1 token/sec)
    await asyncio.sleep(1.1)
    
    # Should have ~1 token now
    allowed, _ = await limiter.check_rate_limit("test_client", cost=1.0)
    assert allowed is True


@pytest.mark.asyncio
async def test_client_status():
    """Test getting client status."""
    limiter = RateLimiter(requests_per_minute=60, burst_size=10)
    
    # Make some requests
    for i in range(3):
        await limiter.check_rate_limit("test_client")
    
    status = limiter.get_client_status("test_client")
    assert status["client_id"] == "test_client"
    assert status["requests_in_window"] == 3
    assert status["tokens_available"] == 7.0
    assert status["metrics"]["allowed"] == 3
    assert status["metrics"]["rejected"] == 0


@pytest.mark.asyncio
async def test_reset_client():
    """Test resetting client rate limits."""
    limiter = RateLimiter(requests_per_minute=60, burst_size=5)
    
    # Use up all tokens
    for i in range(5):
        await limiter.check_rate_limit("test_client")
    
    # Should be blocked
    allowed, _ = await limiter.check_rate_limit("test_client")
    assert allowed is False
    
    # Reset client
    await limiter.reset_client("test_client")
    
    # Should allow requests again
    allowed, _ = await limiter.check_rate_limit("test_client")
    assert allowed is True


@pytest.mark.asyncio
async def test_rate_limit_middleware():
    """Test rate limit middleware."""
    limiter = RateLimiter(requests_per_minute=60, burst_size=10)
    cost_mapping = {
        "expensive_tool": 5.0,
        "cheap_tool": 0.5
    }
    middleware = RateLimitMiddleware(limiter, cost_mapping)
    
    # Test cost mapping
    assert middleware.get_tool_cost("expensive_tool") == 5.0
    assert middleware.get_tool_cost("cheap_tool") == 0.5
    assert middleware.get_tool_cost("unknown_tool") == 1.0
    
    # Test tool limiting
    allowed, _ = await middleware.check_tool_limit("expensive_tool", "session1")
    assert allowed is True
    
    # Second expensive call should exceed burst
    allowed, _ = await middleware.check_tool_limit("expensive_tool", "session1")
    assert allowed is True
    
    # Third should be blocked (10 burst / 5 cost = 2 calls)
    allowed, retry_after = await middleware.check_tool_limit("expensive_tool", "session1")
    assert allowed is False
    assert retry_after > 0


@pytest.mark.asyncio
async def test_rate_limited_decorator():
    """Test the rate_limited decorator."""
    # Create a mock function with rate limiting
    @rate_limited(cost=2.0)
    async def test_function(value: int) -> int:
        return value * 2
    
    # Create a rate limiter and inject it
    limiter = RateLimiter(requests_per_minute=60, burst_size=5)
    test_function._rate_limiter = limiter
    
    # Should work for first few calls
    result = await test_function(5, _session_id="test_session")
    assert result == 10
    
    result = await test_function(10, _session_id="test_session") 
    assert result == 20
    
    # Third call should exceed burst (5 tokens / 2 cost = 2.5 calls)
    with pytest.raises(RateLimitExceeded) as exc_info:
        await test_function(15, _session_id="test_session")
    
    assert exc_info.value.retry_after > 0
    assert "Rate limit exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_rate_limited_decorator_no_limiter():
    """Test rate_limited decorator when no limiter is configured."""
    @rate_limited(cost=1.0)
    async def test_function(value: int) -> int:
        return value * 2
    
    # Should work without rate limiter
    result = await test_function(5)
    assert result == 10