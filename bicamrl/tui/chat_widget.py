"""Chat widget for interacting with Wake agent."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Static


@dataclass
class ChatMessage:
    """Represents a chat message."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class ChatMessageWidget(Static):
    """Widget for displaying a single chat message."""

    def __init__(self, message: ChatMessage, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        """Compose the message widget."""
        # Format timestamp
        time_str = self.message.timestamp.strftime("%H:%M:%S")

        # Create styled content based on role
        if self.message.role == "user":
            content = Panel(
                self.message.content,
                title=f"[cyan]You[/cyan] • {time_str}",
                border_style="cyan",
                padding=(0, 1),
            )
        elif self.message.role == "assistant":
            # Use markdown rendering for assistant messages
            content = Panel(
                RichMarkdown(self.message.content),
                title=f"[green]Wake[/green] • {time_str}",
                border_style="green",
                padding=(0, 1),
            )
        else:  # system
            content = Panel(
                self.message.content,
                title=f"[yellow]System[/yellow] • {time_str}",
                border_style="yellow",
                padding=(0, 1),
            )

        yield Static(content)


class ChatInterface(Vertical):
    """Chat interface for Wake interaction."""

    class MessageSent(Message):
        """Message sent when user sends a chat message."""

        def __init__(self, content: str):
            super().__init__()
            self.content = content

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: List[ChatMessage] = []

    def compose(self) -> ComposeResult:
        """Compose the chat interface."""
        with VerticalScroll(id="chat-messages"):
            yield Static()  # Placeholder for messages

        with Vertical(id="chat-input-container"):
            yield Input(placeholder="Type your message to Wake...", id="chat-input")
            yield Button("Send", variant="primary", id="send-button")

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the chat."""
        message = ChatMessage(
            role=role, content=content, timestamp=datetime.now(), metadata=metadata
        )
        self.messages.append(message)

        # Add to display
        messages_container = self.query_one("#chat-messages", VerticalScroll)
        messages_container.mount(ChatMessageWidget(message))

        # Scroll to bottom
        messages_container.scroll_end(animate=True)

    @on(Input.Submitted, "#chat-input")
    async def on_input_submitted(self, event: Input.Submitted):
        """Handle input submission."""
        if event.value.strip():
            await self._send_message(event.value)
            event.input.clear()

    @on(Button.Pressed, "#send-button")
    async def on_send_pressed(self):
        """Handle send button press."""
        chat_input = self.query_one("#chat-input", Input)
        if chat_input.value.strip():
            await self._send_message(chat_input.value)
            chat_input.clear()

    async def _send_message(self, content: str):
        """Send a message."""
        # Add user message
        self.add_message("user", content)

        # Post event for parent to handle
        self.post_message(self.MessageSent(content))

    def add_system_message(self, content: str):
        """Add a system message."""
        self.add_message("system", content)

    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an assistant message."""
        self.add_message("assistant", content, metadata)
