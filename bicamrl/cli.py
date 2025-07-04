#!/usr/bin/env python3
"""Bicamrl command-line interface."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import click

from .core.memory import Memory

DEFAULT_DB_PATH = ".bicamrl/memory"


@click.group(invoke_without_command=True)
@click.option("--db-path", default=DEFAULT_DB_PATH, help="Path to memory database")
@click.pass_context
def cli(ctx, db_path):
    """Bicamrl - Persistent memory system for AI assistants.

    If no command is specified, launches the interactive TUI.
    """
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path
    ctx.obj["memory"] = Memory(db_path)

    # If no subcommand, launch TUI
    if ctx.invoked_subcommand is None:
        import toml

        from .tui.rust_tui import run_rust_tui

        # Load Mind.toml configuration
        config_path = Path.home() / ".bicamrl" / "Mind.toml"
        config = {}
        if config_path.exists():
            config = toml.load(config_path)  # Let it fail!

        run_rust_tui(db_path, config)


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize memory system for current project."""
    db_path = Path(ctx.obj["db_path"])
    db_path.mkdir(parents=True, exist_ok=True)

    # Create initial CLAUDE.md if it doesn't exist
    claude_md = Path("CLAUDE.md")
    if not claude_md.exists():
        claude_md.write_text(
            """# Bicamrl Context

This file is automatically maintained by Bicamrl.

## Getting Started

The memory system is now active. Use the dashboard to:
- Monitor system activity
- View learned patterns
- Search through interactions
- Analyze memory hierarchy

Run `bicamrl` to launch the interactive interface.

## Recent Activity
*Will be populated after first session*

## Learned Patterns
*Will be populated as patterns are detected*

## Developer Preferences
*Will be populated from feedback*
"""
        )
        click.echo("✓ Created CLAUDE.md")

    click.echo(f"✓ Memory system initialized at {db_path}")
    click.echo("\nNext steps:")
    click.echo("1. Configure your AI tool to use the Bicamrl MCP server")
    click.echo("2. Start working - the system learns automatically")
    click.echo("3. Use `bicamrl` to monitor activity")


@cli.command()
@click.option("--output", "-o", default="memory-export.json", help="Output file")
@click.pass_context
def export(ctx, output):
    """Export memory for sharing or backup."""

    async def _export():
        memory = ctx.obj["memory"]

        data = {
            "exported_at": datetime.now().isoformat(),
            "patterns": await memory.get_all_patterns(),
            "preferences": await memory.get_preferences(),
            "stats": await memory.get_stats(),
        }

        with open(output, "w") as f:
            json.dump(data, f, indent=2)

        click.echo(f"✓ Memory exported to {output}")

    asyncio.run(_export())


@cli.command(name="import")
@click.argument("input_path", type=click.Path(exists=True), required=False)
@click.option("--format", type=click.Choice(["claude-code", "bicamrl"]), default="claude-code")
@click.option("--claude-dir", type=click.Path(exists=True), help="Path to .claude directory")
@click.option("--project-filter", help="Only import conversations from this project path")
@click.option(
    "--current-project/--all-projects",
    default=True,
    help="Import only current project or all projects",
)
@click.option("--merge/--replace", default=True, help="Merge with existing or replace")
@click.pass_context
def import_(ctx, input_path, format, claude_dir, project_filter, current_project, merge):
    """Import memory from Claude Code logs or bicamrl export.

    Examples:
        # Import Claude Code conversations for current project only (default)
        bicamrl import

        # Import all Claude Code conversations from all projects
        bicamrl import --all-projects

        # Import conversations for a specific project
        bicamrl import --project-filter /Users/femtomc/Dev/agents

        # Import specific conversation log
        bicamrl import ~/.claude/projects/-Users-femtomc-Dev-agents/session.jsonl

        # Import from custom Claude directory
        bicamrl import --claude-dir /path/to/.claude
    """

    async def _import(input_path, format, claude_dir, project_filter, current_project, merge):
        import os

        from .core.memory import Memory

        # Initialize memory
        db_path = Path.home() / ".bicamrl" / "memory"
        memory = Memory(db_path)

        if format == "claude-code":
            from .importers.claude_code_importer import ClaudeCodeImporter

            importer = ClaudeCodeImporter(memory)

            if input_path is not None:
                # Import specific file
                input_path = Path(input_path)
                if input_path.is_file() and input_path.suffix == ".jsonl":
                    click.echo(f"Importing Claude Code log: {input_path}")
                    stats = await importer.import_conversation_log(input_path)
                    click.echo(
                        f"✓ Imported: {stats['interactions']} interactions, {stats['actions']} actions"
                    )
                else:
                    click.echo("Error: Input must be a .jsonl file", err=True)
                    ctx.exit(1)
            else:
                # Import directory with filtering
                claude_path = Path(claude_dir) if claude_dir else Path.home() / ".claude"

                # Determine project filter
                if project_filter:
                    # Use explicit filter
                    filter_path = project_filter
                elif current_project:
                    # Use current working directory as filter
                    filter_path = os.getcwd()
                else:
                    # Import all projects
                    filter_path = None

                if filter_path:
                    click.echo(f"Importing Claude Code conversations for project: {filter_path}")
                else:
                    click.echo(f"Importing all Claude Code conversations from: {claude_path}")

                with click.progressbar(length=100, label="Importing conversations") as bar:
                    stats = await importer.import_directory(claude_path, project_filter=filter_path)
                    bar.update(100)

                click.echo("\n✓ Import complete!")
                click.echo(f"  Sessions: {stats['sessions']}")
                click.echo(f"  Interactions: {stats['interactions']}")
                click.echo(f"  Actions: {stats['actions']}")
                click.echo(f"  Patterns detected: {stats['patterns']}")
                if stats.get("skipped_projects", 0) > 0:
                    click.echo(f"  Skipped projects: {stats['skipped_projects']}")
                if stats.get("errors", 0) > 0:
                    click.echo(f"  Errors: {stats['errors']}", fg="yellow")

            # Run memory consolidation after import
            click.echo("\nConsolidating memories...")
            consolidation_stats = await memory.consolidate_memories()
            click.echo(f"✓ Consolidated: {consolidation_stats}")

        else:
            # Original bicamrl format import
            click.echo("Bicamrl format import not yet implemented")

    asyncio.run(_import(input_path, format, claude_dir, project_filter, current_project, merge))


if __name__ == "__main__":
    cli()
