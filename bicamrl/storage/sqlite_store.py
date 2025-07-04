"""SQLite storage backend for memory system."""

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging_config import get_logger
from .json_utils import parse_json_fields, safe_json_dumps, safe_json_loads

logger = get_logger("sqlite_store")


class SQLiteStore:
    """SQLite storage for memory persistence."""

    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self.logger = logger

        logger.info("Initializing SQLiteStore", extra={"db_path": self.db_path})

        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        # Initialize database schema

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Drop old interactions table if it exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='interactions'"
                )
                if cursor.fetchone():
                    logger.info("Dropping old interactions table")
                    conn.execute("DROP TABLE IF EXISTS interactions")
                    conn.execute("DROP INDEX IF EXISTS idx_interactions_session")
                    conn.execute("DROP INDEX IF EXISTS idx_interactions_timestamp")
                    conn.execute("DROP INDEX IF EXISTS idx_interactions_action")
                    conn.execute("DROP INDEX IF EXISTS idx_interactions_file")
                    conn.execute("DROP INDEX IF EXISTS idx_interactions_interaction_id")
                    conn.execute("DROP INDEX IF EXISTS idx_interactions_interaction_id_unique")
                    conn.commit()

            # Add new table for complete interactions
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS complete_interactions (
                    interaction_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    ai_interpretation TEXT,
                    success BOOLEAN DEFAULT 0,
                    feedback_type TEXT,
                    execution_time REAL,
                    tokens_used INTEGER DEFAULT 0,
                    data TEXT NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    pattern_type TEXT NOT NULL,
                    sequence TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    last_seen TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context TEXT,
                    applied BOOLEAN DEFAULT 0
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT,
                    confidence REAL DEFAULT 0.5,
                    source TEXT,
                    updated_at TEXT NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    summary TEXT,
                    files_touched TEXT,
                    patterns_used TEXT,
                    feedback_received INTEGER DEFAULT 0
                )
            """
            )

            # Archive tables for cleanup
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS archived_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_id TEXT,
                    interaction_id TEXT,
                    data TEXT NOT NULL,
                    archived_at TEXT NOT NULL,
                    archive_reason TEXT,
                    consolidated_to TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS archived_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_id TEXT,
                    pattern_name TEXT,
                    pattern_type TEXT,
                    data TEXT NOT NULL,
                    archived_at TEXT NOT NULL,
                    archive_reason TEXT,
                    promoted_to TEXT
                )
            """
            )

            # World model states table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS world_model_states (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    domain TEXT,
                    domain_confidence REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    interaction_count INTEGER DEFAULT 0,
                    entities TEXT NOT NULL,  -- JSON array of entities
                    relations TEXT NOT NULL,  -- JSON array of relations
                    goals TEXT NOT NULL,      -- JSON array of inferred goals
                    metrics TEXT NOT NULL,    -- JSON object with metrics
                    discovered_entity_types TEXT NOT NULL,  -- JSON array
                    discovered_relation_types TEXT NOT NULL, -- JSON array
                    constraints TEXT,         -- JSON object with constraints
                    is_active BOOLEAN DEFAULT 1,
                    merged_from TEXT          -- IDs of world models this was merged from
                )
            """
            )

            # World model snapshots for history
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS world_model_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    world_model_id TEXT NOT NULL,
                    snapshot_time TEXT NOT NULL,
                    snapshot_data TEXT NOT NULL,  -- Full serialized world state
                    trigger_event TEXT,           -- What caused this snapshot
                    FOREIGN KEY (world_model_id) REFERENCES world_model_states(id)
                )
            """
            )

            # Indices for world models
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_world_models_session ON world_model_states(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_world_models_domain ON world_model_states(domain)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_world_models_active ON world_model_states(is_active)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_world_snapshots_model ON world_model_snapshots(world_model_id)"
            )

            # Indices for archive tables
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_archived_interactions_iid ON archived_interactions(interaction_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_archived_interactions_reason ON archived_interactions(archive_reason)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_archived_patterns_type ON archived_patterns(pattern_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_archived_patterns_reason ON archived_patterns(archive_reason)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns(confidence DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_frequency ON patterns(frequency DESC)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_preferences_category ON preferences(category)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_preferences_key ON preferences(key)")

            # Indices for complete_interactions table
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_complete_interactions_session ON complete_interactions(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_complete_interactions_timestamp ON complete_interactions(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_complete_interactions_success ON complete_interactions(success)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_complete_interactions_feedback_type ON complete_interactions(feedback_type)"
            )

            conn.commit()

        except Exception as e:
            logger.error(
                "Database initialization failed",
                extra={"error_type": type(e).__name__, "error": str(e)},
                exc_info=True,
            )
            raise

    async def add_pattern(self, pattern: Dict[str, Any]) -> None:
        """Add a new pattern."""

        def _add():
            pattern_id = pattern.get("id", str(uuid.uuid4()))
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO patterns
                    (id, name, description, pattern_type, sequence, frequency,
                     confidence, last_seen, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        pattern_id,
                        pattern["name"],
                        pattern.get("description", ""),
                        pattern["pattern_type"],
                        safe_json_dumps(pattern.get("sequence", []), default="[]"),
                        pattern.get("frequency", 1),
                        pattern.get("confidence", 0.5),
                        datetime.now().isoformat(),
                        pattern.get("created_at", datetime.now().isoformat()),
                        safe_json_dumps(pattern.get("metadata", {}), default="{}"),
                    ),
                )
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _add)

    async def get_patterns(self, pattern_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all patterns, optionally filtered by type."""

        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if pattern_type:
                    cursor = conn.execute(
                        """
                        SELECT * FROM patterns WHERE pattern_type = ?
                        ORDER BY confidence DESC, frequency DESC
                    """,
                        (pattern_type,),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM patterns
                        ORDER BY confidence DESC, frequency DESC
                    """
                    )

                return [dict(row) for row in cursor.fetchall()]

        rows = await asyncio.get_event_loop().run_in_executor(None, _get)

        # Parse JSON fields safely
        for row in rows:
            parse_json_fields(
                row, ["sequence", "metadata"], defaults={"sequence": [], "metadata": {}}
            )

        return rows

    async def update_pattern_confidence(self, pattern_id: str, confidence: float) -> None:
        """Update pattern confidence score."""

        def _update():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE patterns SET confidence = ?, last_seen = ?
                    WHERE id = ?
                """,
                    (confidence, datetime.now().isoformat(), pattern_id),
                )
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _update)

    async def add_feedback(self, feedback: Dict[str, Any]) -> None:
        """Add feedback."""

        def _add():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO feedback (timestamp, type, message, context)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        feedback["timestamp"],
                        feedback["type"],
                        feedback["message"],
                        safe_json_dumps(feedback.get("context", {}), default="{}"),
                    ),
                )
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _add)

    async def get_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent feedback."""

        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM feedback
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                )

                return [dict(row) for row in cursor.fetchall()]

        rows = await asyncio.get_event_loop().run_in_executor(None, _get)

        # Parse JSON fields safely
        for row in rows:
            parse_json_fields(row, ["context"], defaults={"context": {}})

        return rows

    async def mark_feedback_applied(self, feedback_id: int) -> None:
        """Mark feedback as applied."""

        def _update():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("UPDATE feedback SET applied = 1 WHERE id = ?", (feedback_id,))
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _update)

    async def add_preference(self, preference: Dict[str, Any]) -> None:
        """Add or update a preference."""

        def _add():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO preferences
                    (key, value, category, confidence, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        preference["key"],
                        (
                            safe_json_dumps(preference["value"])
                            if isinstance(preference["value"], (dict, list))
                            else preference["value"]
                        ),
                        preference.get("category", "general"),
                        preference.get("confidence", 0.5),
                        preference.get("source", "unknown"),
                        preference.get("timestamp", datetime.now().isoformat()),
                    ),
                )
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _add)

    async def get_preferences(self) -> List[Dict[str, Any]]:
        """Get all preferences."""

        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM preferences ORDER BY updated_at DESC")
                return [dict(row) for row in cursor.fetchall()]

        rows = await asyncio.get_event_loop().run_in_executor(None, _get)

        # Parse JSON values safely
        for row in rows:
            # Try to parse as JSON, if it fails keep as string
            if (
                row.get("value")
                and isinstance(row["value"], str)
                and row["value"].startswith(("{", "["))
            ):
                row["value"] = safe_json_loads(row["value"], default=row["value"])

        return rows

    async def clear_patterns(self) -> None:
        """Clear all patterns."""

        def _clear():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM patterns")
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _clear)

    async def clear_preferences(self) -> None:
        """Clear all preferences."""

        def _clear():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM preferences")
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _clear)

    async def clear_feedback(self) -> None:
        """Clear all feedback."""

        def _clear():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM feedback")
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _clear)

    async def clear_all(self) -> None:
        """Clear all data."""

        def _clear():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM patterns")
                conn.execute("DELETE FROM feedback")
                conn.execute("DELETE FROM preferences")
                conn.execute("DELETE FROM sessions")
                conn.execute("DELETE FROM complete_interactions")
                conn.execute("DELETE FROM world_model_states")
                conn.execute("DELETE FROM world_model_snapshots")
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _clear)

    async def add_complete_interaction(
        self, interaction_dict: Optional[Dict[str, Any]] = None, **kwargs
    ) -> None:
        """Add a complete interaction to the database.

        Can be called either with a dict or with keyword arguments:
        - add_complete_interaction(interaction_dict)
        - add_complete_interaction(interaction_id=..., session_id=..., ...)
        """
        # If called with keyword arguments, use them
        if interaction_dict is None:
            interaction_dict = kwargs

        def _add():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO complete_interactions
                    (interaction_id, session_id, timestamp, user_query, ai_interpretation,
                     success, feedback_type, execution_time, tokens_used, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        interaction_dict["interaction_id"],
                        interaction_dict["session_id"],
                        interaction_dict["timestamp"],
                        interaction_dict["user_query"],
                        interaction_dict.get("ai_interpretation"),
                        interaction_dict.get("success", False),
                        interaction_dict.get("feedback_type", "none"),
                        interaction_dict.get("execution_time"),
                        interaction_dict.get("tokens_used", 0),
                        safe_json_dumps(interaction_dict, default="{}"),
                    ),
                )
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _add)

    async def get_complete_interactions(
        self,
        session_id: Optional[str] = None,
        interaction_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get complete interactions, optionally filtered by session or interaction_id."""

        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                if interaction_id:
                    cursor = conn.execute(
                        """
                        SELECT data FROM complete_interactions
                        WHERE interaction_id = ?
                    """,
                        (interaction_id,),
                    )
                elif session_id:
                    cursor = conn.execute(
                        """
                        SELECT data FROM complete_interactions
                        WHERE session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """,
                        (session_id, limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT data FROM complete_interactions
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """,
                        (limit,),
                    )

                rows = cursor.fetchall()
                return [json.loads(row["data"]) for row in rows]

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    async def search_interactions_by_query(
        self, query_pattern: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search interactions by user query pattern (simple LIKE search)."""

        def _search():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT data FROM complete_interactions
                    WHERE user_query LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (f"%{query_pattern}%", limit),
                )

                rows = cursor.fetchall()
                return [json.loads(row["data"]) for row in rows]

        return await asyncio.get_event_loop().run_in_executor(None, _search)

    async def archive_patterns(
        self, pattern_ids: List[str], reason: str, promoted_to: str = None
    ) -> int:
        """Archive patterns and remove from active table."""

        def _archive():
            archived_count = 0
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                for pattern_id in pattern_ids:
                    # Get the pattern data
                    cursor = conn.execute("SELECT * FROM patterns WHERE id = ?", (pattern_id,))
                    row = cursor.fetchone()

                    if row:
                        # Archive it
                        conn.execute(
                            """
                            INSERT INTO archived_patterns
                            (original_id, pattern_name, pattern_type, data, archived_at, archive_reason, promoted_to)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                row["id"],
                                row["name"],
                                row["pattern_type"],
                                safe_json_dumps(dict(row), default="{}"),
                                datetime.now().isoformat(),
                                reason,
                                promoted_to,
                            ),
                        )

                        # Delete from active table
                        conn.execute("DELETE FROM patterns WHERE id = ?", (pattern_id,))
                        archived_count += 1

                conn.commit()
                return archived_count

        return await asyncio.get_event_loop().run_in_executor(None, _archive)

    async def get_archived_interaction(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Get an archived interaction by ID."""

        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM archived_interactions WHERE interaction_id = ?",
                    (interaction_id,),
                )
                row = cursor.fetchone()

                if row:
                    result = dict(row)
                    if result.get("data"):
                        result["original_data"] = json.loads(result["data"])
                    return result
                return None

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    async def get_archive_statistics(self) -> Dict[str, Any]:
        """Get statistics about archived data."""

        def _get_stats():
            with sqlite3.connect(self.db_path) as conn:
                # Count archived interactions by reason
                cursor = conn.execute(
                    """
                    SELECT archive_reason, COUNT(*) as count
                    FROM archived_interactions
                    GROUP BY archive_reason
                """
                )
                interaction_stats = {row[0]: row[1] for row in cursor.fetchall()}

                # Count archived patterns by type and reason
                cursor = conn.execute(
                    """
                    SELECT pattern_type, archive_reason, COUNT(*) as count
                    FROM archived_patterns
                    GROUP BY pattern_type, archive_reason
                """
                )
                pattern_stats = {}
                for row in cursor.fetchall():
                    key = f"{row[0]}_{row[1]}"
                    pattern_stats[key] = row[2]

                # Total counts
                cursor = conn.execute("SELECT COUNT(*) FROM archived_interactions")
                total_interactions = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM archived_patterns")
                total_patterns = cursor.fetchone()[0]

                return {
                    "total_archived_interactions": total_interactions,
                    "total_archived_patterns": total_patterns,
                    "interactions_by_reason": interaction_stats,
                    "patterns_by_type_and_reason": pattern_stats,
                }

        return await asyncio.get_event_loop().run_in_executor(None, _get_stats)

    async def vacuum_database(self) -> None:
        """Vacuum the database to reclaim space after deletions."""

        def _vacuum():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")

        await asyncio.get_event_loop().run_in_executor(None, _vacuum)

    # World Model Persistence Methods

    async def add_world_model_state(self, world_state: Dict[str, Any]) -> str:
        """Store a world model state and return its ID."""
        world_model_id = world_state.get("id", str(uuid.uuid4()))

        def _add():
            with sqlite3.connect(self.db_path) as conn:
                # Serialize complex fields to JSON
                entities_json = safe_json_dumps(
                    [
                        {
                            "id": e.id,
                            "type": e.type,
                            "properties": e.properties,
                            "confidence": e.confidence,
                            "first_seen": (
                                e.first_seen.isoformat()
                                if hasattr(e.first_seen, "isoformat")
                                else e.first_seen
                            ),
                            "last_modified": (
                                e.last_modified.isoformat()
                                if hasattr(e.last_modified, "isoformat")
                                else e.last_modified
                            ),
                        }
                        for e in world_state.get("entities", {}).values()
                    ],
                    default="[]",
                )

                relations_json = safe_json_dumps(
                    [
                        {
                            "source_id": r.source_id,
                            "target_id": r.target_id,
                            "type": r.type,
                            "properties": r.properties,
                            "confidence": r.confidence,
                            "observed_count": r.observed_count,
                        }
                        for r in world_state.get("relations", [])
                    ],
                    default="[]",
                )

                goals_json = safe_json_dumps(world_state.get("inferred_goals", []), default="[]")
                metrics_json = safe_json_dumps(world_state.get("metrics", {}), default="{}")
                entity_types_json = safe_json_dumps(
                    list(world_state.get("discovered_entity_types", set())), default="[]"
                )
                relation_types_json = safe_json_dumps(
                    list(world_state.get("discovered_relation_types", set())), default="[]"
                )
                constraints_json = safe_json_dumps(world_state.get("constraints", {}), default="{}")

                now = datetime.now().isoformat()

                conn.execute(
                    """
                    INSERT OR REPLACE INTO world_model_states
                    (id, session_id, domain, domain_confidence, created_at, updated_at,
                     interaction_count, entities, relations, goals, metrics,
                     discovered_entity_types, discovered_relation_types, constraints,
                     is_active, merged_from)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        world_model_id,
                        world_state.get("session_id", "unknown"),
                        world_state.get("domain"),
                        world_state.get("domain_confidence", 0.0),
                        world_state.get("created_at", now),
                        now,
                        world_state.get("interaction_count", 0),
                        entities_json,
                        relations_json,
                        goals_json,
                        metrics_json,
                        entity_types_json,
                        relation_types_json,
                        constraints_json,
                        world_state.get("is_active", True),
                        world_state.get("merged_from"),
                    ),
                )
                conn.commit()

                return world_model_id

        return await asyncio.get_event_loop().run_in_executor(None, _add)

    async def get_world_model_state(self, world_model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a world model state by ID."""

        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM world_model_states WHERE id = ?", (world_model_id,)
                )
                row = cursor.fetchone()

                if row:
                    result = dict(row)
                    # Parse JSON fields
                    result["entities"] = json.loads(result["entities"])
                    result["relations"] = json.loads(result["relations"])
                    result["goals"] = json.loads(result["goals"])
                    result["metrics"] = json.loads(result["metrics"])
                    result["discovered_entity_types"] = set(
                        json.loads(result["discovered_entity_types"])
                    )
                    result["discovered_relation_types"] = set(
                        json.loads(result["discovered_relation_types"])
                    )
                    result["constraints"] = json.loads(result["constraints"])
                    return result
                return None

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    async def get_active_world_models(
        self, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all active world models, optionally filtered by session."""

        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                if session_id:
                    cursor = conn.execute(
                        "SELECT * FROM world_model_states WHERE is_active = 1 AND session_id = ? ORDER BY updated_at DESC",
                        (session_id,),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM world_model_states WHERE is_active = 1 ORDER BY updated_at DESC"
                    )

                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    # Parse JSON fields
                    result["entities"] = json.loads(result["entities"])
                    result["relations"] = json.loads(result["relations"])
                    result["goals"] = json.loads(result["goals"])
                    result["metrics"] = json.loads(result["metrics"])
                    result["discovered_entity_types"] = set(
                        json.loads(result["discovered_entity_types"])
                    )
                    result["discovered_relation_types"] = set(
                        json.loads(result["discovered_relation_types"])
                    )
                    result["constraints"] = json.loads(result["constraints"])
                    results.append(result)

                return results

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    async def get_world_models_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get all world models for a specific domain."""

        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM world_model_states WHERE domain = ? ORDER BY domain_confidence DESC",
                    (domain,),
                )

                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    # Parse JSON fields
                    result["entities"] = json.loads(result["entities"])
                    result["relations"] = json.loads(result["relations"])
                    result["goals"] = json.loads(result["goals"])
                    result["metrics"] = json.loads(result["metrics"])
                    result["discovered_entity_types"] = set(
                        json.loads(result["discovered_entity_types"])
                    )
                    result["discovered_relation_types"] = set(
                        json.loads(result["discovered_relation_types"])
                    )
                    result["constraints"] = json.loads(result["constraints"])
                    results.append(result)

                return results

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    async def add_world_model_snapshot(
        self, world_model_id: str, snapshot_data: Dict[str, Any], trigger_event: str
    ) -> None:
        """Add a snapshot of a world model state."""

        def _add():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO world_model_snapshots
                    (world_model_id, snapshot_time, snapshot_data, trigger_event)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        world_model_id,
                        datetime.now().isoformat(),
                        safe_json_dumps(snapshot_data, default="{}"),
                        trigger_event,
                    ),
                )
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _add)

    async def deactivate_world_model(self, world_model_id: str) -> None:
        """Mark a world model as inactive."""

        def _update():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE world_model_states SET is_active = 0, updated_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), world_model_id),
                )
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _update)
