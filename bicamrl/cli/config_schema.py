"""CLI command to show Mind.toml configuration schema."""

import click

from ..utils.config_validator import ConfigValidator


@click.command()
def config_schema():
    """Show the expected Mind.toml configuration schema."""
    ConfigValidator.print_schema()


if __name__ == "__main__":
    config_schema()
