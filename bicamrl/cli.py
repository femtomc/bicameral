#!/usr/bin/env python3
"""Command-line interface for memory system."""

import asyncio
import click
import json
from pathlib import Path
from datetime import datetime

from .core.memory import Memory
from .core.feedback_processor import FeedbackProcessor

DEFAULT_DB_PATH = ".bicamrl/memory"

@click.group()
@click.option('--db-path', default=DEFAULT_DB_PATH, help='Path to memory database')
@click.pass_context
def cli(ctx, db_path):
    """AI Memory System CLI."""
    ctx.ensure_object(dict)
    ctx.obj['db_path'] = db_path
    ctx.obj['memory'] = Memory(db_path)

@cli.command()
@click.argument('feedback_type', type=click.Choice(['correct', 'prefer', 'pattern']))
@click.argument('message')
@click.pass_context
def feedback(ctx, feedback_type, message):
    """Record feedback for AI learning."""
    async def _feedback():
        memory_manager = ctx.obj['memory']
        processor = FeedbackProcessor(memory_manager)
        result = await processor.process_feedback(feedback_type, message)
        click.echo(f"✓ {result}")
    
    asyncio.run(_feedback())

@cli.command()
@click.argument('action', type=click.Choice(['show', 'search', 'stats', 'clear']))
@click.argument('query', required=False)
@click.pass_context
def memory(ctx, action, query):
    """Query and manage memory system."""
    async def _memory():
        memory_manager = ctx.obj['memory']
        
        if action == 'show':
            if query == 'patterns':
                patterns = await memory_manager.get_all_patterns()
                _display_patterns(patterns)
            elif query == 'preferences':
                preferences = await memory_manager.get_preferences()
                _display_preferences(preferences)
            elif query == 'feedback':
                feedback = await memory_manager.store.get_feedback()
                _display_feedback(feedback)
            else:
                click.echo("Usage: ai-memory show [patterns|preferences|feedback]")
                
        elif action == 'search':
            if not query:
                click.echo("Please provide a search query")
                return
            results = await memory_manager.search(query)
            _display_search_results(results)
            
        elif action == 'stats':
            stats = await memory_manager.get_stats()
            _display_stats(stats)
            
        elif action == 'clear':
            if not query:
                click.echo("Please specify what to clear: patterns, preferences, feedback, or all")
                return
            if query not in ['patterns', 'preferences', 'feedback', 'all']:
                click.echo(f"Invalid clear target: {query}")
                return
                
            if click.confirm(f"Are you sure you want to clear {query}?"):
                await memory_manager.clear_specific(query)
                click.echo(f"✓ Cleared {query}")
    
    asyncio.run(_memory())

def _display_patterns(patterns):
    """Display patterns in a readable format."""
    if not patterns:
        click.echo("No patterns learned yet.")
        return
    
    click.echo("\n📊 Learned Patterns:")
    click.echo("-" * 60)
    
    for p in patterns:
        click.echo(f"\n• {p['name']} (confidence: {p['confidence']:.2f})")
        click.echo(f"  Type: {p['pattern_type']}")
        click.echo(f"  {p.get('description', 'No description')}")
        click.echo(f"  Frequency: {p['frequency']} times")
        if p.get('sequence'):
            click.echo(f"  Sequence: {' → '.join(p['sequence'][:5])}")

def _display_preferences(preferences):
    """Display preferences organized by category."""
    if not preferences:
        click.echo("No preferences recorded yet.")
        return
    
    click.echo("\n⚙️ Developer Preferences:")
    click.echo("-" * 60)
    
    for category, prefs in preferences.items():
        click.echo(f"\n{category.title()}:")
        for key, value in prefs.items():
            click.echo(f"  • {key}: {value}")

def _display_feedback(feedback):
    """Display recent feedback."""
    if not feedback:
        click.echo("No feedback recorded yet.")
        return
    
    click.echo("\n💬 Recent Feedback:")
    click.echo("-" * 60)
    
    for fb in feedback[:10]:  # Show last 10
        timestamp = datetime.fromisoformat(fb['timestamp']).strftime("%Y-%m-%d %H:%M")
        click.echo(f"\n[{timestamp}] {fb['type'].upper()}")
        click.echo(f"  {fb['message']}")
        if fb.get('applied'):
            click.echo("  ✓ Applied")

def _display_search_results(results):
    """Display search results."""
    if not results:
        click.echo("No results found.")
        return
    
    click.echo("\n🔍 Search Results:")
    click.echo("-" * 60)
    
    for r in results[:10]:
        click.echo(f"\n• {r['type'].title()}: {r['name']}")
        click.echo(f"  {r.get('description', '')}")
        if 'file' in r:
            click.echo(f"  File: {r['file']}")
        if 'confidence' in r:
            click.echo(f"  Confidence: {r['confidence']:.2f}")

def _display_stats(stats):
    """Display memory statistics."""
    click.echo("\n📈 Memory Statistics:")
    click.echo("-" * 60)
    click.echo(f"Total Interactions: {stats['total_interactions']}")
    click.echo(f"Patterns Learned: {stats['total_patterns']}")
    click.echo(f"Feedback Received: {stats['total_feedback']}")
    click.echo(f"Active Sessions (24h): {stats['active_sessions']}")
    
    if stats['top_files']:
        click.echo("\nMost Active Files:")
        for f in stats['top_files'][:5]:
            click.echo(f"  • {f}")

@cli.command()
@click.pass_context
def init(ctx):
    """Initialize memory system for current project."""
    db_path = Path(ctx.obj['db_path'])
    db_path.mkdir(parents=True, exist_ok=True)
    
    # Create initial CLAUDE.md if it doesn't exist
    claude_md = Path("CLAUDE.md")
    if not claude_md.exists():
        claude_md.write_text("""# AI Assistant Context

This file is automatically maintained by the AI Memory System.

## Getting Started

The memory system is now active. Use these commands:

- `ai-feedback correct "message"` - Correct mistakes
- `ai-feedback prefer "message"` - Set preferences  
- `ai-feedback pattern "message"` - Teach workflows
- `ai-memory show patterns` - View learned patterns
- `ai-memory show preferences` - View preferences

## Recent Activity
*Will be populated after first session*

## Learned Patterns
*Will be populated as patterns are detected*

## Developer Preferences
*Will be populated from feedback*
""")
        click.echo("✓ Created CLAUDE.md")
    
    click.echo(f"✓ Memory system initialized at {db_path}")
    click.echo("\nNext steps:")
    click.echo("1. Configure your AI tool to use the memory server")
    click.echo("2. Start working - the system learns automatically")
    click.echo("3. Provide feedback to improve understanding")

@cli.command()
@click.option('--output', '-o', default='memory-export.json', help='Output file')
@click.pass_context
def export(ctx, output):
    """Export memory for sharing or backup."""
    async def _export():
        memory_manager = ctx.obj['memory']
        
        data = {
            'exported_at': datetime.now().isoformat(),
            'patterns': await memory_manager.get_all_patterns(),
            'preferences': await memory_manager.get_preferences(),
            'stats': await memory_manager.get_stats()
        }
        
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
        
        click.echo(f"✓ Memory exported to {output}")
    
    asyncio.run(_export())

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--merge/--replace', default=True, help='Merge with existing or replace')
@click.pass_context
def import_(ctx, input_file, merge):
    """Import memory from file (legacy - not supported in new system)."""
    click.echo("Import functionality is not yet supported in the new interaction-based system.")
    click.echo("The system now learns from complete interactions rather than importing patterns.")
    ctx.exit(1)

if __name__ == '__main__':
    cli()