"""Tests for CLI commands."""

import pytest
import json
from pathlib import Path
from click.testing import CliRunner

from bicamrl.cli import cli

@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()

@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory."""
    (tmp_path / ".ai").mkdir()
    return tmp_path

def test_init_command(runner, temp_project):
    """Test initialization command."""
    with runner.isolated_filesystem(temp_dir=temp_project):
        result = runner.invoke(cli, ['init'])
        
        assert result.exit_code == 0
        assert "Memory system initialized" in result.output
        assert Path("CLAUDE.md").exists()
        
        # Check CLAUDE.md content
        content = Path("CLAUDE.md").read_text()
        assert "AI Assistant Context" in content
        assert "Getting Started" in content

def test_feedback_command(runner, temp_project):
    """Test feedback commands."""
    with runner.isolated_filesystem(temp_dir=temp_project):
        # Initialize first
        runner.invoke(cli, ['init'])
        
        # Test correct feedback
        result = runner.invoke(cli, ['feedback', 'correct', 'Use async/await here'])
        assert result.exit_code == 0
        assert "✓" in result.output
        
        # Test prefer feedback  
        result = runner.invoke(cli, ['feedback', 'prefer', 'indent_size: 2'])
        assert result.exit_code == 0
        assert "✓" in result.output
        
        # Test pattern feedback
        result = runner.invoke(cli, ['feedback', 'pattern', 'Always run tests'])
        assert result.exit_code == 0
        assert "✓" in result.output

def test_memory_show_command(runner, temp_project):
    """Test memory show commands."""
    with runner.isolated_filesystem(temp_dir=temp_project):
        runner.invoke(cli, ['init'])
        
        # Add some data first
        runner.invoke(cli, ['feedback', 'prefer', 'test: value'])
        runner.invoke(cli, ['feedback', 'pattern', 'Test pattern'])
        
        # Test show patterns
        result = runner.invoke(cli, ['memory', 'show', 'patterns'])
        assert result.exit_code == 0
        assert "Learned Patterns" in result.output or "No patterns" in result.output
        
        # Test show preferences
        result = runner.invoke(cli, ['memory', 'show', 'preferences'])
        assert result.exit_code == 0
        assert "Developer Preferences" in result.output
        
        # Test show feedback
        result = runner.invoke(cli, ['memory', 'show', 'feedback'])
        assert result.exit_code == 0
        assert "Recent Feedback" in result.output

def test_memory_search_command(runner, temp_project):
    """Test memory search command."""
    with runner.isolated_filesystem(temp_dir=temp_project):
        runner.invoke(cli, ['init'])
        runner.invoke(cli, ['feedback', 'prefer', 'always use pytest'])
        
        # Search for "pytest"
        result = runner.invoke(cli, ['memory', 'search', 'pytest'])
        assert result.exit_code == 0
        assert "Search Results" in result.output or "No results" in result.output

def test_memory_stats_command(runner, temp_project):
    """Test memory stats command."""
    with runner.isolated_filesystem(temp_dir=temp_project):
        runner.invoke(cli, ['init'])
        
        result = runner.invoke(cli, ['memory', 'stats'])
        assert result.exit_code == 0
        assert "Memory Statistics" in result.output
        assert "Total Interactions" in result.output

def test_memory_clear_command(runner, temp_project):
    """Test memory clear command."""
    with runner.isolated_filesystem(temp_dir=temp_project):
        runner.invoke(cli, ['init'])
        
        # Add some data
        runner.invoke(cli, ['feedback', 'prefer', 'test preference'])
        
        # Test clear with confirmation
        result = runner.invoke(cli, ['memory', 'clear', 'preferences'], input='y\n')
        assert result.exit_code == 0
        assert "Cleared preferences" in result.output
        
        # Test clear without target
        result = runner.invoke(cli, ['memory', 'clear'])
        assert "Please specify what to clear" in result.output

def test_export_command(runner, temp_project):
    """Test export command."""
    with runner.isolated_filesystem(temp_dir=temp_project):
        runner.invoke(cli, ['init'])
        
        # Add some data
        runner.invoke(cli, ['feedback', 'prefer', 'exported: preference'])
        
        # Export
        result = runner.invoke(cli, ['export', '-o', 'export.json'])
        assert result.exit_code == 0
        assert "exported to export.json" in result.output
        
        # Check export file
        assert Path("export.json").exists()
        data = json.loads(Path("export.json").read_text())
        assert 'patterns' in data
        assert 'preferences' in data
        assert 'exported_at' in data

def test_import_command(runner, temp_project):
    """Test import command."""
    with runner.isolated_filesystem(temp_dir=temp_project):
        runner.invoke(cli, ['init'])
        
        # Create import file
        import_data = {
            "patterns": [{
                "name": "Imported Pattern",
                "pattern_type": "test",
                "sequence": ["a", "b"],
                "confidence": 0.9
            }],
            "preferences": {
                "testing": {
                    "framework": "pytest"
                }
            }
        }
        
        Path("import.json").write_text(json.dumps(import_data))
        
        # Import with merge - should fail with message about legacy functionality
        result = runner.invoke(cli, ['import-', 'import.json'])
        assert result.exit_code != 0  # Non-zero exit code
        assert "not yet supported in the new interaction-based system" in result.output

def test_import_replace_mode(runner, temp_project):
    """Test import with replace mode."""
    with runner.isolated_filesystem(temp_dir=temp_project):
        runner.invoke(cli, ['init'])
        
        # Create import file
        import_data = {"preferences": {"new": {"key": "value"}}}
        Path("import.json").write_text(json.dumps(import_data))
        
        # Import with replace - should fail with legacy message
        result = runner.invoke(cli, ['import-', 'import.json', '--replace'], input='y\n')
        assert result.exit_code != 0
        assert "not yet supported in the new interaction-based system" in result.output

def test_error_handling(runner, temp_project):
    """Test CLI error handling."""
    with runner.isolated_filesystem(temp_dir=temp_project):
        # Test invalid feedback type
        result = runner.invoke(cli, ['feedback', 'invalid', 'message'])
        assert result.exit_code != 0
        
        # Test import with non-existent file
        result = runner.invoke(cli, ['import-', 'nonexistent.json'])
        assert result.exit_code != 0