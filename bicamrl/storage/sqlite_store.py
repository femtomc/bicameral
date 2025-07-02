"""SQLite storage backend for memory system."""

import json
import sqlite3
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from ..utils.logging_config import get_logger
from ..utils.log_utils import LogContext

logger = get_logger("sqlite_store")

class SQLiteStore:
    """SQLite storage for memory persistence."""
    
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self.logger = logger
        
        logger.info(
            f"Initializing SQLiteStore",
            extra={'db_path': self.db_path}
        )
        
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema."""
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if we need to migrate existing database
                cursor = conn.execute("PRAGMA table_info(interactions)")
                columns = [col[1] for col in cursor.fetchall()]
                
                if 'interaction_id' not in columns and len(columns) > 0:
                    # Migrate existing interactions table
                    logger.warning(
                        "Database migration required: adding interaction_id column",
                        extra={'existing_columns': columns}
                    )
                    
                    with LogContext(logger, "database_migration"):
                        conn.execute('ALTER TABLE interactions ADD COLUMN interaction_id TEXT')
                        
                        # Generate interaction_ids for existing rows
                        conn.execute('''
                            UPDATE interactions 
                            SET interaction_id = timestamp || '_' || action || '_' || id
                            WHERE interaction_id IS NULL
                        ''')
                        
                        # Create unique index
                        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_interactions_interaction_id_unique ON interactions(interaction_id)')
                        conn.commit()
                        
                        logger.info("Database migration completed successfully")
                
                conn.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT UNIQUE,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    action TEXT NOT NULL,
                    file_path TEXT,
                    details TEXT,
                    embeddings BLOB
                )
            ''')
            
            # Add new table for complete interactions
            conn.execute('''
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
            ''')
            
            conn.execute('''
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
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context TEXT,
                    applied BOOLEAN DEFAULT 0
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT,
                    confidence REAL DEFAULT 0.5,
                    source TEXT,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    summary TEXT,
                    files_touched TEXT,
                    patterns_used TEXT,
                    feedback_received INTEGER DEFAULT 0
                )
            ''')
            
            # Archive tables for cleanup
            conn.execute('''
                CREATE TABLE IF NOT EXISTS archived_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_id TEXT,
                    interaction_id TEXT,
                    data TEXT NOT NULL,
                    archived_at TEXT NOT NULL,
                    archive_reason TEXT,
                    consolidated_to TEXT
                )
            ''')
            
            conn.execute('''
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
            ''')
            
            # Create indices for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_interactions_action ON interactions(action)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_interactions_file ON interactions(file_path)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_interactions_interaction_id ON interactions(interaction_id)')
            
            # Indices for archive tables
            conn.execute('CREATE INDEX IF NOT EXISTS idx_archived_interactions_iid ON archived_interactions(interaction_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_archived_interactions_reason ON archived_interactions(archive_reason)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_archived_patterns_type ON archived_patterns(pattern_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_archived_patterns_reason ON archived_patterns(archive_reason)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns(confidence DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_patterns_frequency ON patterns(frequency DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_preferences_category ON preferences(category)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_preferences_key ON preferences(key)')
            
            # Indices for complete_interactions table
            conn.execute('CREATE INDEX IF NOT EXISTS idx_complete_interactions_session ON complete_interactions(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_complete_interactions_timestamp ON complete_interactions(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_complete_interactions_success ON complete_interactions(success)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_complete_interactions_feedback_type ON complete_interactions(feedback_type)')
            
            conn.commit()
        
        except Exception as e:
            logger.error(
                f"Database initialization failed",
                extra={'error_type': type(e).__name__, 'error': str(e)},
                exc_info=True
            )
            raise
    
    async def add_interaction(self, interaction: Dict[str, Any]) -> None:
        """Add an interaction to the database."""
        def _add():
            start_time = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                # Generate interaction_id if not provided
                interaction_id = interaction.get('interaction_id')
                if not interaction_id:
                    interaction_id = f"{interaction['timestamp']}_{interaction['action']}_{str(uuid.uuid4())[:8]}"
                
                logger.debug(
                    f"Storing interaction in SQLite",
                    extra={
                        'interaction_id': interaction_id,
                        'action': interaction['action'],
                        'session_id': interaction.get('session_id'),
                        'file_path': interaction.get('file_path')
                    }
                )
                
                try:
                    conn.execute('''
                        INSERT INTO interactions (interaction_id, timestamp, session_id, action, file_path, details)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        interaction_id,
                        interaction['timestamp'],
                        interaction.get('session_id'),
                        interaction['action'],
                        interaction.get('file_path'),
                        json.dumps(interaction.get('details', {}))
                    ))
                    conn.commit()
                    
                    duration_ms = (time.time() - start_time) * 1000
                    logger.debug(
                        f"Interaction stored successfully",
                        extra={
                            'interaction_id': interaction_id,
                            'duration_ms': duration_ms
                        }
                    )
                    
                except sqlite3.IntegrityError as e:
                    logger.error(
                        f"Failed to store interaction - duplicate ID",
                        extra={
                            'interaction_id': interaction_id,
                            'error': str(e)
                        }
                    )
                    raise
                except Exception as e:
                    logger.error(
                        f"Database error storing interaction",
                        extra={
                            'interaction_id': interaction_id,
                            'error_type': type(e).__name__
                        },
                        exc_info=True
                    )
                    raise
        
        await asyncio.get_event_loop().run_in_executor(None, _add)
    
    async def get_recent_interactions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent interactions."""
        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM interactions
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
        
        rows = await asyncio.get_event_loop().run_in_executor(None, _get)
        
        # Parse JSON fields
        for row in rows:
            if row.get('details'):
                try:
                    row['details'] = json.loads(row['details'])
                except:
                    row['details'] = {}
        
        return rows
    
    async def add_pattern(self, pattern: Dict[str, Any]) -> None:
        """Add a new pattern."""
        def _add():
            pattern_id = pattern.get('id', str(uuid.uuid4()))
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO patterns 
                    (id, name, description, pattern_type, sequence, frequency, 
                     confidence, last_seen, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern_id,
                    pattern['name'],
                    pattern.get('description', ''),
                    pattern['pattern_type'],
                    json.dumps(pattern.get('sequence', [])),
                    pattern.get('frequency', 1),
                    pattern.get('confidence', 0.5),
                    datetime.now().isoformat(),
                    pattern.get('created_at', datetime.now().isoformat()),
                    json.dumps(pattern.get('metadata', {}))
                ))
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(None, _add)
    
    async def get_patterns(self, pattern_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all patterns, optionally filtered by type."""
        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if pattern_type:
                    cursor = conn.execute('''
                        SELECT * FROM patterns WHERE pattern_type = ?
                        ORDER BY confidence DESC, frequency DESC
                    ''', (pattern_type,))
                else:
                    cursor = conn.execute('''
                        SELECT * FROM patterns
                        ORDER BY confidence DESC, frequency DESC
                    ''')
                
                return [dict(row) for row in cursor.fetchall()]
        
        rows = await asyncio.get_event_loop().run_in_executor(None, _get)
        
        # Parse JSON fields
        for row in rows:
            if row.get('sequence'):
                try:
                    row['sequence'] = json.loads(row['sequence'])
                except:
                    row['sequence'] = []
            if row.get('metadata'):
                try:
                    row['metadata'] = json.loads(row['metadata'])
                except:
                    row['metadata'] = {}
        
        return rows
    
    async def update_pattern_confidence(self, pattern_id: str, confidence: float) -> None:
        """Update pattern confidence score."""
        def _update():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE patterns SET confidence = ?, last_seen = ?
                    WHERE id = ?
                ''', (confidence, datetime.now().isoformat(), pattern_id))
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(None, _update)
    
    async def add_feedback(self, feedback: Dict[str, Any]) -> None:
        """Add feedback."""
        def _add():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO feedback (timestamp, type, message, context)
                    VALUES (?, ?, ?, ?)
                ''', (
                    feedback['timestamp'],
                    feedback['type'],
                    feedback['message'],
                    json.dumps(feedback.get('context', {}))
                ))
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(None, _add)
    
    async def get_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent feedback."""
        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM feedback
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
        
        rows = await asyncio.get_event_loop().run_in_executor(None, _get)
        
        # Parse JSON fields
        for row in rows:
            if row.get('context'):
                try:
                    row['context'] = json.loads(row['context'])
                except:
                    row['context'] = {}
        
        return rows
    
    async def mark_feedback_applied(self, feedback_id: int) -> None:
        """Mark feedback as applied."""
        def _update():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('UPDATE feedback SET applied = 1 WHERE id = ?', (feedback_id,))
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(None, _update)
    
    async def add_preference(self, preference: Dict[str, Any]) -> None:
        """Add or update a preference."""
        def _add():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO preferences 
                    (key, value, category, confidence, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    preference['key'],
                    json.dumps(preference['value']) if isinstance(preference['value'], (dict, list)) else preference['value'],
                    preference.get('category', 'general'),
                    preference.get('confidence', 0.5),
                    preference.get('source', 'unknown'),
                    preference.get('timestamp', datetime.now().isoformat())
                ))
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(None, _add)
    
    async def get_preferences(self) -> List[Dict[str, Any]]:
        """Get all preferences."""
        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('SELECT * FROM preferences ORDER BY updated_at DESC')
                return [dict(row) for row in cursor.fetchall()]
        
        rows = await asyncio.get_event_loop().run_in_executor(None, _get)
        
        # Parse JSON values
        for row in rows:
            try:
                row['value'] = json.loads(row['value'])
            except:
                pass  # Keep as string if not JSON
        
        return rows
    
    async def clear_patterns(self) -> None:
        """Clear all patterns."""
        def _clear():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM patterns')
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(None, _clear)
    
    async def clear_preferences(self) -> None:
        """Clear all preferences."""
        def _clear():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM preferences')
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(None, _clear)
    
    async def clear_feedback(self) -> None:
        """Clear all feedback."""
        def _clear():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM feedback')
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(None, _clear)
    
    async def clear_all(self) -> None:
        """Clear all data."""
        def _clear():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM interactions')
                conn.execute('DELETE FROM patterns')
                conn.execute('DELETE FROM feedback')
                conn.execute('DELETE FROM preferences')
                conn.execute('DELETE FROM sessions')
                conn.execute('DELETE FROM complete_interactions')
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(None, _clear)
    
    async def add_complete_interaction(self, interaction_dict: Dict[str, Any]) -> None:
        """Add a complete interaction to the database."""
        def _add():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO complete_interactions 
                    (interaction_id, session_id, timestamp, user_query, ai_interpretation,
                     success, feedback_type, execution_time, tokens_used, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    interaction_dict['interaction_id'],
                    interaction_dict['session_id'],
                    interaction_dict['timestamp'],
                    interaction_dict['user_query'],
                    interaction_dict.get('ai_interpretation'),
                    interaction_dict.get('success', False),
                    interaction_dict.get('feedback_type', 'none'),
                    interaction_dict.get('execution_time'),
                    interaction_dict.get('tokens_used', 0),
                    json.dumps(interaction_dict)
                ))
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(None, _add)
    
    async def get_complete_interactions(self, 
                                      session_id: Optional[str] = None,
                                      limit: int = 100) -> List[Dict[str, Any]]:
        """Get complete interactions, optionally filtered by session."""
        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if session_id:
                    cursor = conn.execute('''
                        SELECT data FROM complete_interactions
                        WHERE session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (session_id, limit))
                else:
                    cursor = conn.execute('''
                        SELECT data FROM complete_interactions
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (limit,))
                
                rows = cursor.fetchall()
                return [json.loads(row['data']) for row in rows]
        
        return await asyncio.get_event_loop().run_in_executor(None, _get)
    
    async def search_interactions_by_query(self, query_pattern: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search interactions by user query pattern (simple LIKE search)."""
        def _search():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT data FROM complete_interactions
                    WHERE user_query LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (f'%{query_pattern}%', limit))
                
                rows = cursor.fetchall()
                return [json.loads(row['data']) for row in rows]
        
        return await asyncio.get_event_loop().run_in_executor(None, _search)
    
    # Archive methods for cleanup
    async def archive_interactions(self, interaction_ids: List[str], reason: str, consolidated_to: str = None) -> int:
        """Archive interactions and remove from active table."""
        def _archive():
            archived_count = 0
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                for interaction_id in interaction_ids:
                    # Get the interaction data
                    cursor = conn.execute(
                        'SELECT * FROM interactions WHERE interaction_id = ?',
                        (interaction_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        # Archive it
                        conn.execute('''
                            INSERT INTO archived_interactions 
                            (original_id, interaction_id, data, archived_at, archive_reason, consolidated_to)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            row['id'],
                            row['interaction_id'],
                            json.dumps(dict(row)),
                            datetime.now().isoformat(),
                            reason,
                            consolidated_to
                        ))
                        
                        # Delete from active table
                        conn.execute(
                            'DELETE FROM interactions WHERE interaction_id = ?',
                            (interaction_id,)
                        )
                        archived_count += 1
                
                conn.commit()
                return archived_count
        
        return await asyncio.get_event_loop().run_in_executor(None, _archive)
    
    async def archive_patterns(self, pattern_ids: List[str], reason: str, promoted_to: str = None) -> int:
        """Archive patterns and remove from active table."""
        def _archive():
            archived_count = 0
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                for pattern_id in pattern_ids:
                    # Get the pattern data
                    cursor = conn.execute(
                        'SELECT * FROM patterns WHERE id = ?',
                        (pattern_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        # Archive it
                        conn.execute('''
                            INSERT INTO archived_patterns 
                            (original_id, pattern_name, pattern_type, data, archived_at, archive_reason, promoted_to)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            row['id'],
                            row['name'],
                            row['pattern_type'],
                            json.dumps(dict(row)),
                            datetime.now().isoformat(),
                            reason,
                            promoted_to
                        ))
                        
                        # Delete from active table
                        conn.execute(
                            'DELETE FROM patterns WHERE id = ?',
                            (pattern_id,)
                        )
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
                    'SELECT * FROM archived_interactions WHERE interaction_id = ?',
                    (interaction_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    result = dict(row)
                    if result.get('data'):
                        result['original_data'] = json.loads(result['data'])
                    return result
                return None
        
        return await asyncio.get_event_loop().run_in_executor(None, _get)
    
    async def get_archive_statistics(self) -> Dict[str, Any]:
        """Get statistics about archived data."""
        def _get_stats():
            with sqlite3.connect(self.db_path) as conn:
                # Count archived interactions by reason
                cursor = conn.execute('''
                    SELECT archive_reason, COUNT(*) as count
                    FROM archived_interactions
                    GROUP BY archive_reason
                ''')
                interaction_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Count archived patterns by type and reason
                cursor = conn.execute('''
                    SELECT pattern_type, archive_reason, COUNT(*) as count
                    FROM archived_patterns
                    GROUP BY pattern_type, archive_reason
                ''')
                pattern_stats = {}
                for row in cursor.fetchall():
                    key = f"{row[0]}_{row[1]}"
                    pattern_stats[key] = row[2]
                
                # Total counts
                cursor = conn.execute('SELECT COUNT(*) FROM archived_interactions')
                total_interactions = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT COUNT(*) FROM archived_patterns')
                total_patterns = cursor.fetchone()[0]
                
                return {
                    'total_archived_interactions': total_interactions,
                    'total_archived_patterns': total_patterns,
                    'interactions_by_reason': interaction_stats,
                    'patterns_by_type_and_reason': pattern_stats
                }
        
        return await asyncio.get_event_loop().run_in_executor(None, _get_stats)
    
    async def vacuum_database(self) -> None:
        """Vacuum the database to reclaim space after deletions."""
        def _vacuum():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('VACUUM')
        
        await asyncio.get_event_loop().run_in_executor(None, _vacuum)