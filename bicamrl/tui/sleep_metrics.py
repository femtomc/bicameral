"""Sleep metrics visualization widget."""

from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Optional

from rich.panel import Panel
from textual.app import ComposeResult
from textual.containers import Container, Grid, Horizontal
from textual.widgets import DataTable, Label, Sparkline, Static


class MetricCard(Container):
    """A card showing a single metric."""

    def __init__(
        self,
        title: str,
        value: str = "N/A",
        subtitle: str = "",
        trend: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.trend = trend

    def compose(self) -> ComposeResult:
        """Compose the metric card."""
        trend_str = ""
        if self.trend is not None:
            if self.trend > 0:
                trend_str = f" [green]↑{self.trend:+.1%}[/green]"
            elif self.trend < 0:
                trend_str = f" [red]↓{self.trend:+.1%}[/red]"
            else:
                trend_str = " [dim]→0.0%[/dim]"

        content = f"[bold]{self.value}[/bold]{trend_str}\n[dim]{self.subtitle}[/dim]"

        yield Static(Panel(content, title=self.title, border_style="blue"), classes="metric-card")

    def update_metric(self, value: str, subtitle: str = "", trend: Optional[float] = None):
        """Update the metric value."""
        self.value = value
        self.subtitle = subtitle
        self.trend = trend
        self.refresh()


class SleepMetrics(Container):
    """Sleep system metrics dashboard."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_history = deque(maxlen=60)  # Last 60 observations
        self.insight_counts = defaultdict(int)
        self.role_usage = defaultdict(int)
        self.llm_latencies = deque(maxlen=100)
        self.success_rates = deque(maxlen=50)

    def compose(self) -> ComposeResult:
        """Compose the Sleep metrics interface."""
        yield Label("[bold]Sleep System Metrics[/bold]", id="sleep-title")

        # Top metrics cards
        with Grid(id="sleep-metrics-grid"):
            yield MetricCard("Observations/min", "0", id="obs-rate-card")
            yield MetricCard("Active Role", "None", id="active-role-card")
            yield MetricCard("Insights", "0", "Total generated", id="insights-card")
            yield MetricCard("Avg Latency", "0ms", id="latency-card")
            yield MetricCard("Success Rate", "N/A", id="success-card")
            yield MetricCard("Memory Efficiency", "0%", id="memory-card")

        # Activity sparkline
        yield Label("Observation Activity (last 60s)")
        yield Sparkline([], id="obs-sparkline")

        # Insights breakdown
        yield Label("Insights by Type")
        yield DataTable(id="insights-table", cursor_type="row")

        # Role performance
        yield Label("Role Performance")
        yield DataTable(id="role-table", cursor_type="row")

        # LLM provider status
        yield Label("LLM Provider Status")
        with Horizontal(id="llm-status"):
            yield Static("[dim]No providers active[/dim]", id="llm-status-text")

    async def on_mount(self):
        """Initialize tables when mounted."""
        # Initialize insights table
        insights_table = self.query_one("#insights-table", DataTable)
        insights_table.add_columns("Type", "Count", "Last Seen", "Avg Confidence")

        # Initialize role table
        role_table = self.query_one("#role-table", DataTable)
        role_table.add_columns("Role", "Activations", "Success Rate", "Avg Response Time")

    def update_observation_rate(self, observations_per_minute: float):
        """Update observation rate metric."""
        card = self.query_one("#obs-rate-card", MetricCard)

        # Calculate trend
        if len(self.observation_history) > 10:
            old_rate = sum(self.observation_history[:10]) / 10
            new_rate = sum(self.observation_history[-10:]) / 10
            trend = (new_rate - old_rate) / max(old_rate, 1)
        else:
            trend = None

        card.update_metric(f"{observations_per_minute:.1f}", "per minute", trend)

        # Update sparkline
        sparkline = self.query_one("#obs-sparkline", Sparkline)
        sparkline.data = list(self.observation_history)

    def update_active_role(self, role_name: Optional[str], confidence: Optional[float] = None):
        """Update active role display."""
        card = self.query_one("#active-role-card", MetricCard)
        if role_name:
            subtitle = f"Confidence: {confidence:.1%}" if confidence else ""
            card.update_metric(role_name, subtitle)
        else:
            card.update_metric("None", "No role active")

    def update_insights(self, insight_type: str, confidence: float):
        """Update insights metrics."""
        self.insight_counts[insight_type] += 1

        # Update card
        total = sum(self.insight_counts.values())
        card = self.query_one("#insights-card", MetricCard)
        card.update_metric(str(total), f"{len(self.insight_counts)} types")

        # Update table
        self._update_insights_table()

    def update_llm_latency(self, provider: str, latency_ms: float):
        """Update LLM latency metrics."""
        self.llm_latencies.append(latency_ms)

        # Update card
        if self.llm_latencies:
            avg_latency = sum(self.llm_latencies) / len(self.llm_latencies)
            card = self.query_one("#latency-card", MetricCard)

            # Calculate trend
            if len(self.llm_latencies) > 10:
                old_avg = sum(list(self.llm_latencies)[:10]) / 10
                new_avg = sum(list(self.llm_latencies)[-10:]) / 10
                trend = (new_avg - old_avg) / max(old_avg, 1)
            else:
                trend = None

            card.update_metric(f"{avg_latency:.0f}ms", f"{provider}", trend)

    def update_success_rate(self, success: bool):
        """Update success rate metric."""
        self.success_rates.append(1.0 if success else 0.0)

        if self.success_rates:
            rate = sum(self.success_rates) / len(self.success_rates)
            card = self.query_one("#success-card", MetricCard)

            # Calculate trend
            if len(self.success_rates) > 10:
                old_rate = sum(list(self.success_rates)[:10]) / 10
                new_rate = sum(list(self.success_rates)[-10:]) / 10
                trend = new_rate - old_rate
            else:
                trend = None

            card.update_metric(f"{rate:.1%}", f"Last {len(self.success_rates)}", trend)

    def update_memory_efficiency(self, active: int, working: int, episodic: int, semantic: int):
        """Update memory efficiency metric."""
        total = active + working + episodic + semantic
        if total > 0:
            # Higher level memories are more efficient
            efficiency = (working * 0.3 + episodic * 0.5 + semantic * 1.0) / total
            card = self.query_one("#memory-card", MetricCard)
            card.update_metric(
                f"{efficiency:.1%}", f"A:{active} W:{working} E:{episodic} S:{semantic}"
            )

    def update_role_performance(self, role: str, success: bool, response_time_ms: float):
        """Update role performance metrics."""
        self.role_usage[role] += 1
        self._update_role_table()

    def update_llm_status(self, providers: Dict[str, str]):
        """Update LLM provider status."""
        status_text = self.query_one("#llm-status-text", Static)
        if providers:
            status_parts = []
            for name, status in providers.items():
                if status == "active":
                    status_parts.append(f"[green]●[/green] {name}")
                elif status == "error":
                    status_parts.append(f"[red]●[/red] {name}")
                else:
                    status_parts.append(f"[yellow]●[/yellow] {name}")
            status_text.update(" ".join(status_parts))
        else:
            status_text.update("[dim]No providers active[/dim]")

    def _update_insights_table(self):
        """Update the insights breakdown table."""
        table = self.query_one("#insights-table", DataTable)
        table.clear()

        for insight_type, count in sorted(
            self.insight_counts.items(), key=lambda x: x[1], reverse=True
        ):
            table.add_row(
                insight_type.replace("_", " ").title(),
                str(count),
                datetime.now().strftime("%H:%M:%S"),
                "85%",  # Placeholder confidence
            )

    def _update_role_table(self):
        """Update the role performance table."""
        table = self.query_one("#role-table", DataTable)
        table.clear()

        for role, count in sorted(self.role_usage.items(), key=lambda x: x[1], reverse=True):
            table.add_row(
                role,
                str(count),
                "92%",  # Placeholder success rate
                "145ms",  # Placeholder response time
            )

    def add_observation(self):
        """Record a new observation."""
        # Add to history for rate calculation
        self.observation_history.append(1)

        # Calculate rate
        if len(self.observation_history) > 0:
            rate = sum(self.observation_history) * 60 / len(self.observation_history)
            self.update_observation_rate(rate)
