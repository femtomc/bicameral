"""Test Claude Code SDK integration."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from bicamrl.sleep.llm_providers import ClaudeCodeProvider, create_llm_providers


class TestClaudeCodeProvider:
    """Test Claude Code provider functionality."""

    @pytest.mark.asyncio
    async def test_claude_code_initialization_success(self):
        """Test successful initialization when dependencies are available."""
        with patch('shutil.which', return_value='/usr/local/bin/claude-code'):
            with patch.dict('sys.modules', {'claude_code_sdk': MagicMock()}):
                provider = ClaudeCodeProvider()
                assert provider is not None
                assert provider.api_key is None  # Claude Code doesn't need API key

    def test_claude_code_initialization_missing_sdk(self):
        """Test initialization fails when SDK is not installed."""
        with patch.dict('sys.modules', {'claude_code_sdk': None}):
            with pytest.raises(ImportError, match="claude-code-sdk not installed"):
                ClaudeCodeProvider()

    def test_claude_code_initialization_missing_cli(self):
        """Test initialization fails when CLI is not installed."""
        with patch('shutil.which', return_value=None):
            with patch.dict('sys.modules', {'claude_code_sdk': MagicMock()}):
                with pytest.raises(RuntimeError, match="Claude Code CLI not found"):
                    ClaudeCodeProvider()

    @pytest.mark.asyncio
    async def test_claude_code_analyze(self):
        """Test analyze method with mocked Claude Code SDK."""
        # Mock the SDK
        mock_message = Mock()
        mock_message.content = "This is the analysis result"

        async def mock_query(prompt, options):
            yield mock_message

        with patch('shutil.which', return_value='/usr/local/bin/claude-code'):
            with patch.dict('sys.modules', {'claude_code_sdk': MagicMock()}):
                with patch('claude_code_sdk.query', side_effect=mock_query):
                    provider = ClaudeCodeProvider()
                    result = await provider.analyze("Test prompt", system="Test system")
                    assert result == "This is the analysis result"

    @pytest.mark.asyncio
    async def test_claude_code_generate(self):
        """Test generate method with mocked Claude Code SDK."""
        # Mock the SDK
        mock_message1 = Mock()
        mock_message1.content = "Part 1 "
        mock_message2 = Mock()
        mock_message2.content = "Part 2"

        async def mock_query(prompt, options):
            yield mock_message1
            yield mock_message2

        with patch('shutil.which', return_value='/usr/local/bin/claude-code'):
            with patch.dict('sys.modules', {'claude_code_sdk': MagicMock()}):
                with patch('claude_code_sdk.query', side_effect=mock_query):
                    provider = ClaudeCodeProvider()
                    result = await provider.generate("Generate something")
                    assert result == "Part 1 Part 2"

    def test_create_llm_providers_with_claude_code(self):
        """Test that create_llm_providers can create Claude Code provider."""
        config = {
            'claude_code': {
                'type': 'claude_code',
                'temperature': 0.5
            }
        }

        with patch('shutil.which', return_value='/usr/local/bin/claude-code'):
            with patch.dict('sys.modules', {'claude_code_sdk': MagicMock()}):
                providers = create_llm_providers(config)
                assert 'claude_code' in providers
                assert isinstance(providers['claude_code'], ClaudeCodeProvider)

    def test_create_llm_providers_claude_code_failure(self):
        """Test that provider creation continues when Claude Code fails."""
        config = {
            'claude_code': {
                'type': 'claude_code'
            },
            'openai': {
                'api_key': 'test-key'
            }
        }

        with patch('shutil.which', return_value=None):
            providers = create_llm_providers(config)
            # Claude Code should fail, but other providers should still be created
            assert 'claude_code' not in providers
            assert 'openai' in providers
            assert 'mock' in providers  # Mock is always included as fallback


@pytest.mark.integration
class TestClaudeCodeIntegration:
    """Integration tests that require actual Claude Code CLI."""

    @pytest.mark.asyncio
    async def test_real_claude_code_query(self):
        """Test actual Claude Code query (requires CLI installed)."""
        try:
            provider = ClaudeCodeProvider()
            result = await provider.generate("Say 'Hello from Claude Code test'")
            assert "Hello" in result
            assert len(result) > 0
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Claude Code not available: {e}")
