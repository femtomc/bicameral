"""Health check functionality for Bicamrl server."""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from .utils.logging_config import get_logger

logger = get_logger("health")


class HealthChecker:
    """Manages health checks for server components."""
    
    def __init__(self):
        self.startup_time = datetime.now()
        self.checks = {}
        
    async def check_memory(self, memory) -> Dict[str, Any]:
        """Check memory system health."""
        try:
            start = time.time()
            # Try to get stats
            stats = await memory.get_stats()
            latency_ms = (time.time() - start) * 1000
            
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "total_interactions": stats.get("total_interactions", 0),
                "total_patterns": stats.get("total_patterns", 0),
                "message": "Memory system operational"
            }
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Memory system error"
            }
    
    async def check_storage(self, memory) -> Dict[str, Any]:
        """Check storage backend health."""
        try:
            # Check SQLite
            start = time.time()
            db_path = Path(memory.db_path) / "memory.db"
            sqlite_exists = db_path.exists()
            sqlite_size = db_path.stat().st_size if sqlite_exists else 0
            
            result = {
                "sqlite": {
                    "status": "healthy" if sqlite_exists else "unhealthy",
                    "exists": sqlite_exists,
                    "size_mb": round(sqlite_size / 1024 / 1024, 2),
                }
            }
            
            # Check hybrid store if available
            if memory.hybrid_store:
                result["vector_store"] = {
                    "status": "healthy",
                    "backend": memory.hybrid_store.vector_store.__class__.__name__,
                    "has_embeddings": memory.hybrid_store.embeddings is not None
                }
            
            latency_ms = (time.time() - start) * 1000
            result["latency_ms"] = round(latency_ms, 2)
            result["status"] = "healthy" if sqlite_exists else "unhealthy"
            
            return result
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Storage check error"
            }
    
    async def check_sleep_layer(self, sleep_layer) -> Dict[str, Any]:
        """Check Sleep layer health."""
        if not sleep_layer:
            return {
                "status": "disabled",
                "message": "Sleep layer not configured"
            }
            
        try:
            return {
                "status": "healthy" if sleep_layer.is_running else "stopped",
                "is_running": sleep_layer.is_running,
                "queue_size": sleep_layer.observation_queue.qsize() if hasattr(sleep_layer, 'observation_queue') else 0,
                "insights_cached": len(sleep_layer.insights_cache) if hasattr(sleep_layer, 'insights_cache') else 0,
                "providers": list(sleep_layer.llms.keys()) if hasattr(sleep_layer, 'llms') else [],
                "message": "Sleep layer operational" if sleep_layer.is_running else "Sleep layer stopped"
            }
        except Exception as e:
            logger.error(f"Sleep layer health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Sleep layer error"
            }
    
    async def check_all(self, memory, sleep_layer, pattern_detector, feedback_processor) -> Dict[str, Any]:
        """Run all health checks."""
        start = time.time()
        
        # Run checks in parallel
        memory_check, storage_check, sleep_check = await asyncio.gather(
            self.check_memory(memory),
            self.check_storage(memory),
            self.check_sleep_layer(sleep_layer),
            return_exceptions=True
        )
        
        # Handle any exceptions from gather
        if isinstance(memory_check, Exception):
            memory_check = {"status": "error", "error": str(memory_check)}
        if isinstance(storage_check, Exception):
            storage_check = {"status": "error", "error": str(storage_check)}
        if isinstance(sleep_check, Exception):
            sleep_check = {"status": "error", "error": str(sleep_check)}
        
        # Component status
        components = {
            "memory": memory_check,
            "storage": storage_check,
            "sleep_layer": sleep_check,
            "pattern_detector": {
                "status": "healthy" if pattern_detector else "not_initialized",
                "message": "Pattern detector operational" if pattern_detector else "Not initialized"
            },
            "feedback_processor": {
                "status": "healthy" if feedback_processor else "not_initialized",
                "message": "Feedback processor operational" if feedback_processor else "Not initialized"
            }
        }
        
        # Determine overall status
        statuses = [comp.get("status", "unknown") for comp in components.values()]
        if all(s in ["healthy", "disabled"] for s in statuses):
            overall_status = "healthy"
        elif any(s in ["unhealthy", "error"] for s in statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        # Build response
        uptime_seconds = (datetime.now() - self.startup_time).total_seconds()
        check_duration_ms = (time.time() - start) * 1000
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(uptime_seconds, 2),
            "check_duration_ms": round(check_duration_ms, 2),
            "components": components,
            "version": "1.0.0",  # You might want to read this from package
        }


# Global health checker instance
health_checker = HealthChecker()