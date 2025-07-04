"""Bicamrl TUI entry point."""

import sys
from pathlib import Path

import toml

from ..config_loader import get_memory_path
from ..config_loader import load_config as load_json_config
from ..utils.config_validator import ConfigValidator
from .rust_tui import run_rust_tui


def load_mind_config():
    """Load Mind.toml configuration."""
    # First check local .bicamrl/Mind.toml
    local_config_path = Path.cwd() / ".bicamrl" / "Mind.toml"
    if local_config_path.exists():
        try:
            config = toml.load(local_config_path)
            # Validate the configuration
            errors = ConfigValidator.validate_mind_config(config)
            if errors:
                print("Configuration errors in local Mind.toml:", file=sys.stderr)
                for error in errors:
                    print(f"  - {error}", file=sys.stderr)
                print("\nPlease fix the configuration errors above.", file=sys.stderr)
                sys.exit(1)
            return config
        except Exception as e:
            print(f"Error: Failed to load local Mind.toml: {e}", file=sys.stderr)
            sys.exit(1)

    # Then check home directory
    config_path = Path.home() / ".bicamrl" / "Mind.toml"
    if config_path.exists():
        try:
            config = toml.load(config_path)
            # Validate the configuration
            errors = ConfigValidator.validate_mind_config(config)
            if errors:
                print("Configuration errors in Mind.toml:", file=sys.stderr)
                for error in errors:
                    print(f"  - {error}", file=sys.stderr)
                print("\nPlease fix the configuration errors above.", file=sys.stderr)
                sys.exit(1)
            return config
        except Exception as e:
            print(f"Error: Failed to load Mind.toml: {e}", file=sys.stderr)
            sys.exit(1)

    # No configuration found
    print("Error: No Mind.toml configuration found!", file=sys.stderr)
    print("Please create one at either:", file=sys.stderr)
    print("  - .bicamrl/Mind.toml (in current directory)", file=sys.stderr)
    print("  - ~/.bicamrl/Mind.toml (in home directory)", file=sys.stderr)
    print("\nRun 'bicamrl config-schema' to see the expected format.", file=sys.stderr)
    sys.exit(1)


def main():
    """Main entry point for Bicamrl TUI."""
    # Load configuration to get database path
    config = load_json_config()
    db_path = str(get_memory_path(config))

    # Load Mind.toml for Sleep configuration
    mind_config = load_mind_config()

    # Ensure the database directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        run_rust_tui(db_path, mind_config)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error running TUI: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
