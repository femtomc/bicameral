"""Tests for health check functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from bicamrl.health import HealthChecker


@pytest.mark.asyncio
async def test_health_checker_initialization():
    """Test health checker initializes correctly."""
    checker = HealthChecker()
    assert isinstance(checker.startup_time, datetime)
    assert checker.checks == {}


@pytest.mark.asyncio
async def test_check_memory_healthy():
    """Test memory health check when healthy."""
    checker = HealthChecker()
    memory = Mock()
    memory.get_stats = AsyncMock(return_value={
        "total_interactions": 100,
        "total_patterns": 10
    })
    
    result = await checker.check_memory(memory)
    
    assert result["status"] == "healthy"
    assert "latency_ms" in result
    assert result["total_interactions"] == 100
    assert result["total_patterns"] == 10
    assert result["message"] == "Memory system operational"


@pytest.mark.asyncio
async def test_check_memory_unhealthy():
    """Test memory health check when unhealthy."""
    checker = HealthChecker()
    memory = Mock()
    memory.get_stats = AsyncMock(side_effect=Exception("DB Error"))
    
    result = await checker.check_memory(memory)
    
    assert result["status"] == "unhealthy"
    assert result["error"] == "DB Error"
    assert result["message"] == "Memory system error"


@pytest.mark.asyncio
async def test_check_storage_healthy():
    """Test storage health check when healthy."""
    checker = HealthChecker()
    memory = Mock()
    memory.db_path = "/tmp/test"
    memory.hybrid_store = Mock()
    memory.hybrid_store.vector_store = Mock()
    memory.hybrid_store.vector_store.__class__.__name__ = "TestVectorStore"
    memory.hybrid_store.embeddings = Mock()
    
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = 1024 * 1024  # 1MB
        
        result = await checker.check_storage(memory)
        
        assert result["status"] == "healthy"
        assert result["sqlite"]["status"] == "healthy"
        assert result["sqlite"]["exists"] is True
        assert result["sqlite"]["size_mb"] == 1.0
        assert result["vector_store"]["status"] == "healthy"
        assert result["vector_store"]["backend"] == "TestVectorStore"
        assert result["vector_store"]["has_embeddings"] is True


@pytest.mark.asyncio
async def test_check_sleep_layer_disabled():
    """Test sleep layer check when disabled."""
    checker = HealthChecker()
    
    result = await checker.check_sleep_layer(None)
    
    assert result["status"] == "disabled"
    assert result["message"] == "Sleep layer not configured"


@pytest.mark.asyncio
async def test_check_sleep_layer_healthy():
    """Test sleep layer check when healthy."""
    checker = HealthChecker()
    sleep_layer = Mock()
    sleep_layer.is_running = True
    sleep_layer.observation_queue = Mock()
    sleep_layer.observation_queue.qsize = Mock(return_value=5)
    sleep_layer.insights_cache = [1, 2, 3]
    sleep_layer.llms = {"openai": Mock(), "claude": Mock()}
    
    result = await checker.check_sleep_layer(sleep_layer)
    
    assert result["status"] == "healthy"
    assert result["is_running"] is True
    assert result["queue_size"] == 5
    assert result["insights_cached"] == 3
    assert result["providers"] == ["openai", "claude"]
    assert result["message"] == "Sleep layer operational"


@pytest.mark.asyncio
async def test_check_all_healthy():
    """Test overall health check when all components healthy."""
    checker = HealthChecker()
    
    # Mock components
    memory = Mock()
    memory.get_stats = AsyncMock(return_value={"total_interactions": 100})
    memory.db_path = "/tmp/test"
    memory.hybrid_store = None
    
    sleep_layer = Mock()
    sleep_layer.is_running = True
    sleep_layer.observation_queue = Mock()
    sleep_layer.observation_queue.qsize = Mock(return_value=0)
    sleep_layer.insights_cache = []
    sleep_layer.llms = {}
    
    pattern_detector = Mock()
    feedback_processor = Mock()
    
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = 1024 * 1024
        
        result = await checker.check_all(memory, sleep_layer, pattern_detector, feedback_processor)
        
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "uptime_seconds" in result
        assert "check_duration_ms" in result
        assert result["version"] == "1.0.0"
        
        # Check components
        assert result["components"]["memory"]["status"] == "healthy"
        assert result["components"]["storage"]["status"] == "healthy"
        assert result["components"]["sleep_layer"]["status"] == "healthy"
        assert result["components"]["pattern_detector"]["status"] == "healthy"
        assert result["components"]["feedback_processor"]["status"] == "healthy"


@pytest.mark.asyncio
async def test_check_all_unhealthy():
    """Test overall health check when some components unhealthy."""
    checker = HealthChecker()
    
    # Mock components with failures
    memory = Mock()
    memory.get_stats = AsyncMock(side_effect=Exception("DB Error"))
    memory.db_path = "/tmp/test"
    
    result = await checker.check_all(memory, None, None, None)
    
    assert result["status"] == "unhealthy"
    assert result["components"]["memory"]["status"] == "unhealthy"
    assert result["components"]["sleep_layer"]["status"] == "disabled"
    assert result["components"]["pattern_detector"]["status"] == "not_initialized"
    assert result["components"]["feedback_processor"]["status"] == "not_initialized"