"""Bicamrl Terminal User Interface - unified dashboard for system visibility."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from collections import defaultdict

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Button, Static, Label, Tree, DataTable,
    TabbedContent, TabPane, Log, Input, Select, Sparkline,
    LoadingIndicator, Placeholder, Pretty, RichLog
)
from textual.reactive import reactive
from textual.worker import Worker
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from ..core.memory import Memory
from ..storage.sqlite_store import SQLiteStore
from ..storage.hybrid_store import HybridStore


class BicamrlTUI(App):
    """Bicamrl Terminal User Interface."""
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #main-container {
        height: 100%;
    }
    
    #stats-grid {
        layout: grid;
        grid-size: 3 2;
        grid-gutter: 1;
        margin: 1;
        height: 8;
    }
    
    .stat-box {
        border: solid green;
        content-align: center middle;
        padding: 1;
    }
    
    #memory-tree {
        border: solid blue;
        height: 100%;
    }
    
    #pattern-table {
        border: solid yellow;
        height: 100%;
    }
    
    #activity-log {
        border: solid cyan;
        height: 100%;
        overflow-y: scroll;
    }
    
    #search-results {
        border: solid magenta;
        height: 100%;
        overflow-y: scroll;
    }
    
    .detail-view {
        border: solid white;
        padding: 1;
        height: 100%;
        overflow-y: scroll;
    }
    
    Input {
        dock: top;
        margin: 1;
    }
    
    .tab-content {
        height: 100%;
    }
    
    #world-model-status {
        dock: top;
        height: 3;
        border: solid white;
        content-align: center middle;
        margin: 1;
    }
    
    #entity-container {
        width: 50%;
        border: solid cyan;
        height: 100%;
        margin: 1;
    }
    
    #goal-container {
        width: 50%;
        border: solid green;
        height: 100%;
        margin: 1;
    }
    
    #world-model-tree {
        height: 100%;
    }
    
    #world-model-goals {
        height: 100%;
        padding: 1;
    }
    
    #world-model-proposals {
        border: solid yellow;
        height: 30%;
        margin: 1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("tab", "focus_next", "Next Tab"),
        Binding("shift+tab", "focus_previous", "Previous Tab"),
        Binding("?", "help", "Help"),
    ]
    
    def __init__(self, db_path: str = None):
        super().__init__()
        self.db_path = db_path or str(Path.home() / ".bicamrl" / "memory")
        self.memory: Optional[Memory] = None
        self.selected_interaction_id: Optional[str] = None
        self.selected_pattern_id: Optional[str] = None
        self.refresh_worker: Optional[Worker] = None
        
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header(show_clock=True)
        
        with TabbedContent(initial="overview", id="main-container"):
            # Overview Tab
            with TabPane("Overview", id="overview"):
                yield from self._compose_overview()
            
            # Memory Inspector Tab
            with TabPane("Memory", id="memory"):
                yield from self._compose_memory_inspector()
            
            # Patterns Tab
            with TabPane("Patterns", id="patterns"):
                yield from self._compose_patterns()
            
            # World Model Tab
            with TabPane("World Model", id="world-model"):
                yield from self._compose_world_model()
            
            # Activity Monitor Tab
            with TabPane("Activity", id="activity"):
                yield from self._compose_activity()
            
            # Search Tab
            with TabPane("Search", id="search"):
                yield from self._compose_search()
        
        yield Footer()
    
    def _compose_overview(self) -> ComposeResult:
        """Compose the overview tab."""
        with Vertical(classes="tab-content"):
            # Stats grid
            with Container(id="stats-grid"):
                yield Static("Loading...", classes="stat-box", id="stat-interactions")
                yield Static("Loading...", classes="stat-box", id="stat-patterns")
                yield Static("Loading...", classes="stat-box", id="stat-success")
                yield Static("Loading...", classes="stat-box", id="stat-active")
                yield Static("Loading...", classes="stat-box", id="stat-tokens")
                yield Static("Loading...", classes="stat-box", id="stat-sessions")
            
            # Activity sparkline
            yield Sparkline([], id="activity-sparkline")
            
            # Recent events log
            yield RichLog(highlight=True, markup=True, id="events-log")
    
    def _compose_memory_inspector(self) -> ComposeResult:
        """Compose the memory inspector tab."""
        with Horizontal(classes="tab-content"):
            # Memory hierarchy tree
            with Vertical(id="memory-tree"):
                yield Label("Memory Hierarchy")
                yield Tree("Memory", id="memory-tree-widget")
            
            # Detail view
            yield Pretty(None, classes="detail-view", id="memory-detail")
    
    def _compose_patterns(self) -> ComposeResult:
        """Compose the patterns tab."""
        with Vertical(classes="tab-content"):
            # Controls
            with Horizontal():
                yield Select(
                    [("all", "All Types"), ("workflow", "Workflow"), 
                     ("file_access", "File Access"), ("error", "Error")],
                    id="pattern-type-filter",
                    value="all"
                )
                yield Button("Analyze Recent", id="analyze-patterns")
            
            # Pattern table
            yield DataTable(id="pattern-table")
            
            # Pattern detail
            yield Pretty(None, classes="detail-view", id="pattern-detail")
    
    def _compose_world_model(self) -> ComposeResult:
        """Compose the world model tab."""
        with Vertical(classes="tab-content"):
            # World model status
            yield Static("Loading world model...", id="world-model-status")
            
            with Horizontal():
                # Entity graph
                with Vertical(id="entity-container"):
                    yield Label("Entities & Relations")
                    yield Tree("World Model", id="world-model-tree")
                
                # Goal understanding
                with Vertical(id="goal-container"):
                    yield Label("Inferred Goals")
                    yield Pretty(None, id="world-model-goals")
            
            # Proposals
            yield Label("Goal-Directed Proposals")
            yield DataTable(id="world-model-proposals")
    
    def _compose_activity(self) -> ComposeResult:
        """Compose the activity monitor tab."""
        with Vertical(classes="tab-content"):
            yield Label("Real-time Activity Monitor")
            yield Log(highlight=True, id="activity-log", auto_scroll=True)
    
    def _compose_search(self) -> ComposeResult:
        """Compose the search tab."""
        with Vertical(classes="tab-content"):
            yield Input(placeholder="Search interactions, patterns, or files...", id="search-input")
            yield RichLog(highlight=True, markup=True, id="search-results")
    
    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        self.memory = Memory(self.db_path)
        
        # Initialize data tables
        pattern_table = self.query_one("#pattern-table", DataTable)
        pattern_table.add_columns("Pattern", "Type", "Frequency", "Confidence", "Last Seen")
        
        proposal_table = self.query_one("#world-model-proposals", DataTable)
        proposal_table.add_columns("Type", "Description", "Confidence", "Rationale")
        
        # Start refresh worker
        self.refresh_data()
        
        # Set up auto-refresh
        self.set_interval(5, self.refresh_data)
    
    @work(exclusive=True)
    async def refresh_data(self) -> None:
        """Refresh all data displays."""
        await self._update_overview()
        await self._update_memory_tree()
        await self._update_patterns()
        await self._update_world_model()
        await self._update_activity()
    
    async def _update_overview(self) -> None:
        """Update overview statistics."""
        stats = await self.memory.get_stats()
        
        # Update stat boxes
        self.query_one("#stat-interactions").update(
            f"[bold]Interactions[/bold]\n{stats['total_interactions']}"
        )
        self.query_one("#stat-patterns").update(
            f"[bold]Patterns[/bold]\n{stats['total_patterns']}"
        )
        
        # Calculate success rate
        recent = await self.memory.store.get_recent_complete_interactions(hours=24)
        if recent:
            success_rate = sum(1 for i in recent if i['success']) / len(recent)
            self.query_one("#stat-success").update(
                f"[bold]Success Rate[/bold]\n{success_rate:.1%}"
            )
        else:
            self.query_one("#stat-success").update(
                f"[bold]Success Rate[/bold]\nN/A"
            )
        
        # Active memories
        active_count = await self.memory.store.count_recent_interactions(hours=1)
        self.query_one("#stat-active").update(
            f"[bold]Active[/bold]\n{active_count}"
        )
        
        # Token usage
        total_tokens = sum(i.get('tokens_used', 0) for i in recent)
        self.query_one("#stat-tokens").update(
            f"[bold]Tokens (24h)[/bold]\n{total_tokens:,}"
        )
        
        # Active sessions
        self.query_one("#stat-sessions").update(
            f"[bold]Sessions[/bold]\n{stats['active_sessions']}"
        )
        
        # Update activity sparkline
        hourly_data = self._calculate_hourly_activity(recent)
        sparkline = self.query_one("#activity-sparkline", Sparkline)
        sparkline.data = list(hourly_data.values())
        
        # Add recent events to log
        events_log = self.query_one("#events-log", RichLog)
        for interaction in recent[:5]:
            timestamp = datetime.fromisoformat(interaction['timestamp']).strftime("%H:%M")
            success = "✅" if interaction['success'] else "❌"
            events_log.write(f"[dim]{timestamp}[/dim] {success} {interaction['user_query'][:50]}...")
    
    async def _update_memory_tree(self) -> None:
        """Update memory hierarchy tree."""
        tree = self.query_one("#memory-tree-widget", Tree)
        tree.clear()
        
        # Get counts for each level
        active_count = await self.memory.store.count_recent_interactions(hours=1)
        working_patterns = await self.memory.store.get_patterns(pattern_type='working_memory')
        episodic_patterns = await self.memory.store.get_patterns(pattern_type='episodic_memory') 
        semantic_patterns = await self.memory.store.get_patterns(pattern_type='semantic_knowledge')
        
        # Build tree
        root = tree.root
        
        # Active memory
        active_node = root.add(f"[green]Active Memory ({active_count})[/green]")
        recent = await self.memory.store.get_recent_complete_interactions(limit=5)
        for interaction in recent:
            node = active_node.add(
                f"{interaction['user_query'][:40]}...",
                data={"type": "interaction", "id": interaction['interaction_id']}
            )
        
        # Working memory
        working_node = root.add(f"[yellow]Working Memory ({len(working_patterns)})[/yellow]")
        for pattern in working_patterns[:5]:
            node = working_node.add(
                pattern['name'][:40],
                data={"type": "pattern", "id": pattern['id']}
            )
        
        # Episodic memory
        episodic_node = root.add(f"[cyan]Episodic Memory ({len(episodic_patterns)})[/cyan]")
        for pattern in episodic_patterns[:5]:
            node = episodic_node.add(
                pattern['name'][:40],
                data={"type": "pattern", "id": pattern['id']}
            )
        
        # Semantic memory
        semantic_node = root.add(f"[magenta]Semantic Knowledge ({len(semantic_patterns)})[/magenta]")
        for pattern in semantic_patterns[:5]:
            node = semantic_node.add(
                pattern['name'][:40],
                data={"type": "pattern", "id": pattern['id']}
            )
        
        root.expand_all()
    
    async def _update_patterns(self) -> None:
        """Update patterns table."""
        pattern_type = self.query_one("#pattern-type-filter", Select).value
        
        # Get patterns
        if pattern_type == "all":
            patterns = await self.memory.store.get_all_patterns()
        else:
            patterns = await self.memory.store.get_patterns(pattern_type=pattern_type)
        
        # Sort by confidence
        patterns.sort(key=lambda p: p['confidence'], reverse=True)
        
        # Update table
        table = self.query_one("#pattern-table", DataTable)
        table.clear()
        
        for pattern in patterns[:20]:
            # Format last seen
            last_seen = pattern.get('last_seen', 'Unknown')
            if last_seen != 'Unknown':
                try:
                    last_seen_dt = datetime.fromisoformat(last_seen)
                    last_seen = last_seen_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            table.add_row(
                pattern['name'][:40],
                pattern['pattern_type'],
                str(pattern['frequency']),
                f"{pattern['confidence']:.2%}",
                last_seen,
                key=pattern['id']
            )
    
    async def _update_world_model(self) -> None:
        """Update world model visualization."""
        try:
            # Get world models from semantic memory
            patterns = await self.memory.store.get_patterns(pattern_type="semantic_knowledge")
            world_models = [p for p in patterns if p.get("metadata", {}).get("is_world_model")]
            
            if not world_models:
                self.query_one("#world-model-status").update("No world models found yet")
                return
            
            # Use the most recent world model
            latest_world = max(world_models, key=lambda w: w.get("created_at", ""))
            world_data = latest_world.get("sequence", {})
            if isinstance(world_data, str):
                world_data = json.loads(world_data)
            
            # Update status
            status_text = f"Domain: {world_data.get('domain', 'unknown')} | "
            status_text += f"Entities: {len(world_data.get('entities', {}))} | "
            status_text += f"Relations: {len(world_data.get('relations', []))}"
            self.query_one("#world-model-status").update(status_text)
            
            # Update entity tree
            tree = self.query_one("#world-model-tree", Tree)
            tree.clear()
            root = tree.root
            
            # Add entities
            entities_node = root.add("[cyan]Entities[/cyan]")
            for entity_id, entity in world_data.get("entities", {}).items():
                entity_node = entities_node.add(
                    f"[yellow]{entity['type']}[/yellow]: {entity_id}",
                    data={"type": "entity", "data": entity}
                )
                # Add properties
                for prop, value in entity.get("properties", {}).items():
                    entity_node.add(f"{prop}: {value}")
            
            # Add relations
            relations_node = root.add("[magenta]Relations[/magenta]")
            for relation in world_data.get("relations", [])[:10]:  # Limit to 10
                rel_text = f"{relation['source_id']} --{relation['type']}--> {relation['target_id']}"
                relations_node.add(rel_text, data={"type": "relation", "data": relation})
            
            root.expand_all()
            
            # Update goals
            goals = world_data.get("inferred_goals", [])
            if goals:
                latest_goal = goals[-1]  # Most recent goal
                goal_display = {
                    "type": latest_goal.get("type", "unknown"),
                    "description": latest_goal.get("description", "No description"),
                    "achieved": latest_goal.get("achieved", False),
                    "timestamp": latest_goal.get("timestamp", "unknown")
                }
                self.query_one("#world-model-goals").update(goal_display)
            else:
                self.query_one("#world-model-goals").update({"status": "No goals inferred yet"})
            
            # Update proposals (would come from Sleep if integrated)
            proposal_table = self.query_one("#world-model-proposals", DataTable)
            proposal_table.clear()
            
            # For now, show some example proposals based on world state
            if world_data.get("domain") == "coding":
                proposal_table.add_row(
                    "NEXT_ACTION",
                    "Add test coverage for recent changes",
                    "0.8",
                    "Test files missing for modified code"
                )
            elif world_data.get("domain") == "writing":
                proposal_table.add_row(
                    "GOAL_CLARIFICATION",
                    "Confirm target audience for document",
                    "0.7",
                    "Audience affects writing style"
                )
                
        except Exception as e:
            self.query_one("#world-model-status").update(f"Error: {str(e)}")
    
    async def _update_activity(self) -> None:
        """Update activity monitor."""
        log = self.query_one("#activity-log", Log)
        
        # Get recent interactions
        recent = await self.memory.store.get_recent_complete_interactions(limit=10)
        
        for interaction in recent:
            timestamp = datetime.fromisoformat(interaction['timestamp'])
            time_str = timestamp.strftime("%H:%M:%S")
            
            # Format the log entry
            success = "✅" if interaction['success'] else "❌"
            data = json.loads(interaction['data']) if isinstance(interaction['data'], str) else interaction['data']
            action_count = len(data.get('actions_taken', []))
            
            log.write_line(
                f"{time_str} {success} Query: {interaction['user_query'][:50]}... "
                f"({action_count} actions, {interaction.get('execution_time', 0):.1f}s)"
            )
    
    def _calculate_hourly_activity(self, interactions: List[Dict[str, Any]]) -> Dict[int, int]:
        """Calculate hourly activity for sparkline."""
        hourly = defaultdict(int)
        now = datetime.now()
        
        # Initialize last 24 hours
        for i in range(24):
            hour = (now - timedelta(hours=i)).hour
            hourly[hour] = 0
        
        # Count interactions
        for interaction in interactions:
            timestamp = datetime.fromisoformat(interaction['timestamp'])
            if now - timestamp < timedelta(hours=24):
                hourly[timestamp.hour] += 1
        
        # Return in chronological order
        return {k: hourly[k] for k in sorted(hourly.keys())}
    
    @on(Tree.NodeSelected)
    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        if not event.node.data:
            return
        
        data = event.node.data
        detail_view = self.query_one("#memory-detail", Pretty)
        
        if data["type"] == "interaction":
            # Load interaction details
            interactions = await self.memory.store.get_complete_interactions(
                interaction_id=data["id"]
            )
            if interactions:
                detail_view.update(interactions[0])
        elif data["type"] == "pattern":
            # Load pattern details
            patterns = await self.memory.store.get_patterns(pattern_id=data["id"])
            if patterns:
                detail_view.update(patterns[0])
    
    @on(DataTable.RowSelected)
    async def on_pattern_selected(self, event: DataTable.RowSelected) -> None:
        """Handle pattern selection from table."""
        pattern_id = event.row_key.value
        patterns = await self.memory.store.get_patterns(pattern_id=pattern_id)
        
        if patterns:
            detail_view = self.query_one("#pattern-detail", Pretty)
            detail_view.update(patterns[0])
    
    @on(Button.Pressed, "#analyze-patterns")
    async def analyze_patterns(self) -> None:
        """Run pattern analysis."""
        # This would trigger the pattern detection logic
        self.notify("Running pattern analysis...", severity="information")
        # Implementation would go here
        await self.refresh_data()
        self.notify("Pattern analysis complete!", severity="success")
    
    @on(Input.Submitted, "#search-input")
    async def search(self, event: Input.Submitted) -> None:
        """Handle search submission."""
        query = event.value
        results_log = self.query_one("#search-results", RichLog)
        results_log.clear()
        
        if not query:
            return
        
        results_log.write(f"[bold]Searching for: {query}[/bold]\n")
        
        # Search interactions
        results_log.write("\n[yellow]Interactions:[/yellow]")
        interactions = await self.memory.search(query)
        
        if interactions:
            for interaction in interactions[:10]:
                results_log.write(
                    f"  • {interaction['timestamp']}: {interaction['query'][:60]}..."
                )
        else:
            results_log.write("  No matching interactions found")
        
        # Search patterns
        results_log.write("\n[cyan]Patterns:[/cyan]")
        all_patterns = await self.memory.store.get_all_patterns()
        matching_patterns = [
            p for p in all_patterns 
            if query.lower() in p['name'].lower() or 
               query.lower() in p.get('description', '').lower()
        ]
        
        if matching_patterns:
            for pattern in matching_patterns[:5]:
                results_log.write(
                    f"  • {pattern['name']} ({pattern['pattern_type']}, "
                    f"confidence: {pattern['confidence']:.1%})"
                )
        else:
            results_log.write("  No matching patterns found")
    
    def action_refresh(self) -> None:
        """Refresh action."""
        self.refresh_data()
        self.notify("Data refreshed", severity="information")
    
    def action_help(self) -> None:
        """Show help."""
        self.push_screen("help")


def run_tui(db_path: Optional[str] = None):
    """Run the Bicamrl TUI."""
    app = BicamrlTUI(db_path)
    app.run()


if __name__ == "__main__":
    run_tui()