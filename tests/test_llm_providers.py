"""Test LLM provider implementations."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bicamrl.sleep.llm_providers import (
    ClaudeLLMProvider,
    LocalLLMProvider,
    MockLLMProvider,
    OpenAILLMProvider,
    create_llm_providers,
)

# Check if optional dependencies are available
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class TestMockLLMProvider:
    """Test mock LLM provider."""
    
    @pytest.mark.asyncio
    async def test_mock_analyze(self):
        """Test mock analyze functionality."""
        provider = MockLLMProvider()
        
        # Test pattern analysis
        result = await provider.analyze("Analyze these patterns: [1, 2, 3]")
        assert "patterns" in result.lower()
        
        # Test JSON response
        data = json.loads(result)
        assert "patterns" in data
        assert len(data["patterns"]) == 1
        assert data["patterns"][0]["type"] == "workflow"
        assert data["patterns"][0]["confidence"] == 0.8
    
    @pytest.mark.asyncio
    async def test_mock_generate(self):
        """Test mock generate functionality."""
        provider = MockLLMProvider()
        
        result = await provider.generate("Generate a story")
        assert "mock enhanced prompt" in result.lower()
        assert len(result) > 10


@pytest.mark.skipif(not HAS_OPENAI, reason="OpenAI library not installed")
class TestOpenAILLMProvider:
    """Test OpenAI LLM provider."""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client."""
        with patch("openai.AsyncOpenAI") as mock:
            client = MagicMock()
            mock.return_value = client
            
            # Mock chat completion
            completion = MagicMock()
            completion.choices = [MagicMock(message=MagicMock(content="Test response"))]
            
            client.chat.completions.create = AsyncMock(return_value=completion)
            
            yield client
    
    @pytest.mark.asyncio
    async def test_openai_analyze(self, mock_openai):
        """Test OpenAI analyze functionality."""
        config = {
            "api_key": "test-key",
            "model": "gpt-4",
            "max_tokens": 1000
        }
        
        provider = OpenAILLMProvider(api_key="test-key", config=config)
        
        result = await provider.analyze("Analyze this text")
        assert result == "Test response"
        
        # Verify API call
        mock_openai.chat.completions.create.assert_called_once()
        call_args = mock_openai.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4"
        assert call_args.kwargs["max_tokens"] == 1000
    
    @pytest.mark.asyncio
    async def test_openai_generate_with_temperature(self, mock_openai):
        """Test OpenAI generate with temperature."""
        config = {
            "api_key": "test-key",
            "model": "gpt-4",
            "temperature": 0.8
        }
        
        provider = OpenAILLMProvider(api_key="test-key", config=config)
        
        result = await provider.generate("Generate creative text", temperature=0.9)
        assert result == "Test response"
        
        # Should use provided temperature
        call_args = mock_openai.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.9
    
    @pytest.mark.asyncio
    async def test_openai_retry_logic(self, mock_openai):
        """Test retry logic on API errors."""
        # First call fails, second succeeds
        mock_openai.chat.completions.create.side_effect = [
            Exception("API Error"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Success"))])
        ]
        
        config = {
            "api_key": "test-key",
            "model": "gpt-4",
            "max_retries": 3
        }
        
        provider = OpenAILLMProvider(api_key="test-key", config=config)
        
        result = await provider.analyze("Test with retry")
        assert result == "Success"
        assert mock_openai.chat.completions.create.call_count == 2


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic library not installed")
class TestClaudeLLMProvider:
    """Test Claude LLM provider."""
    
    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client."""
        with patch("anthropic.AsyncAnthropic") as mock:
            client = MagicMock()
            mock.return_value = client
            
            # Mock message response
            message = MagicMock()
            message.content = [MagicMock(text="Claude response")]
            
            client.messages.create = AsyncMock(return_value=message)
            
            yield client
    
    @pytest.mark.asyncio
    async def test_claude_analyze(self, mock_anthropic):
        """Test Claude analyze functionality."""
        config = {
            "api_key": "test-key",
            "model": "claude-3-opus-20240229",
            "max_tokens": 2000
        }
        
        provider = ClaudeLLMProvider(api_key="test-key", config=config)
        
        result = await provider.analyze("Analyze this")
        assert result == "Claude response"
        
        # Verify API call
        mock_anthropic.messages.create.assert_called_once()
        call_args = mock_anthropic.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-3-opus-20240229"
        assert call_args.kwargs["max_tokens"] == 2000
    
    @pytest.mark.asyncio
    async def test_claude_system_message(self, mock_anthropic):
        """Test Claude with system message."""
        config = {
            "api_key": "test-key",
            "model": "claude-3-opus-20240229"
        }
        
        provider = ClaudeLLMProvider(api_key="test-key", config=config)
        
        await provider.generate("Test", system="You are a helpful assistant")
        
        call_args = mock_anthropic.messages.create.call_args
        assert call_args.kwargs["system"] == "You are a helpful assistant"


class TestLocalLLMProvider:
    """Test local LLM provider (LM Studio)."""
    
    @pytest.fixture
    def mock_aiohttp(self):
        """Mock aiohttp for API calls."""
        with patch("aiohttp.ClientSession") as mock:
            client = MagicMock()
            mock.return_value = client
            
            # Mock response for local LLM (Ollama format)
            response = MagicMock()
            response.json = AsyncMock(return_value={
                "response": "Local LLM response"
            })
            response.raise_for_status = MagicMock()
            
            # Mock context manager
            context = AsyncMock()
            context.__aenter__ = AsyncMock(return_value=response)
            context.__aexit__ = AsyncMock(return_value=None)
            
            session = AsyncMock()
            session.post = MagicMock(return_value=context)
            
            client.__aenter__ = AsyncMock(return_value=session)
            client.__aexit__ = AsyncMock(return_value=None)
            
            yield mock
    
    @pytest.mark.asyncio
    async def test_local_llm_analyze(self, mock_aiohttp):
        """Test local LLM analyze functionality."""
        config = {
            "base_url": "http://localhost:1234/v1",
            "model": "local-model"
        }
        
        provider = LocalLLMProvider(config=config)
        
        result = await provider.analyze("Analyze this")
        assert result == "Local LLM response"
        
        # Verify API call
        mock_aiohttp.assert_called()
    
    @pytest.mark.asyncio
    async def test_local_llm_timeout(self, mock_aiohttp):
        """Test local LLM with timeout."""
        # Make the mock raise a timeout error
        mock_aiohttp.side_effect = asyncio.TimeoutError()
        
        config = {
            "base_url": "http://localhost:1234/v1",
            "model": "local-model",
            "timeout": 10
        }
        
        provider = LocalLLMProvider(config=config)
        
        with pytest.raises(asyncio.TimeoutError):
            await provider.analyze("Test timeout")


class TestCreateLLMProviders:
    """Test LLM provider factory function."""
    
    def test_create_providers_with_openai(self):
        """Test creating providers with OpenAI config."""
        config = {
            "openai": {
                "api_key": "test-key",
                "model": "gpt-4"
            }
        }
        
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            providers = create_llm_providers(config)
        
        # Should have analyzer role mapped to OpenAI
        assert "analyzer" in providers
        assert isinstance(providers["analyzer"], OpenAILLMProvider)
    
    def test_create_providers_with_claude(self):
        """Test creating providers with Claude config."""
        config = {
            "claude": {
                "api_key": "test-key",
                "model": "claude-3-opus-20240229"
            }
        }
        
        providers = create_llm_providers(config)
        
        # Should have generator role mapped to Claude (default when Claude is available)
        assert "generator" in providers
        assert isinstance(providers["generator"], ClaudeLLMProvider)
    
    def test_create_providers_with_lmstudio(self):
        """Test creating providers with LM Studio config."""
        config = {
            "lmstudio": {
                "api_base": "http://localhost:1234/v1",
                "model": "local-model"
            }
        }
        
        providers = create_llm_providers(config)
        
        # LM Studio uses OpenAILLMProvider with custom base URL
        # Since it's the only provider, mock will be used for roles
        assert "analyzer" in providers
        assert isinstance(providers["analyzer"], MockLLMProvider)
    
    def test_create_providers_with_mock(self):
        """Test creating providers with mock."""
        config = {
            "mock": {
                "type": "mock"
            }
        }
        
        providers = create_llm_providers(config)
        
        # The function returns role mappings, not provider names
        assert "analyzer" in providers
        assert "generator" in providers
        assert isinstance(providers["analyzer"], MockLLMProvider)
        assert isinstance(providers["generator"], MockLLMProvider)
    
    def test_create_providers_with_env_vars(self):
        """Test that providers can be created with placeholder API keys."""
        config = {
            "openai": {
                "api_key": "${OPENAI_API_KEY}",  # This is still a valid string
                "model": "gpt-4"
            }
        }
        
        providers = create_llm_providers(config)
        
        # Should have analyzer role (OpenAI is assigned to analyzer by default)
        assert "analyzer" in providers
        assert isinstance(providers["analyzer"], OpenAILLMProvider)
    
    def test_create_providers_skip_invalid(self):
        """Test skipping invalid provider configs."""
        config = {
            "openai": {
                # Missing api_key
                "model": "gpt-4"
            },
            "mock": {
                "type": "mock"
            }
        }
        
        providers = create_llm_providers(config)
        
        # Should use mock for all roles since openai is invalid
        assert "analyzer" in providers
        assert isinstance(providers["analyzer"], MockLLMProvider)


class TestLLMProviderIntegration:
    """Test LLM provider integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_provider_role_assignment(self):
        """Test using different providers for different roles."""
        config = {
            "openai": {
                "api_key": "test-key-1",
                "model": "gpt-4"
            },
            "claude": {
                "api_key": "test-key-2", 
                "model": "claude-3-opus-20240229"
            }
        }
        
        providers = create_llm_providers(config)
        
        # Check role assignments
        # Default: OpenAI -> analyzer, Claude -> generator
        analyzer = providers.get("analyzer")
        generator = providers.get("generator")
        
        assert analyzer is not None
        assert generator is not None
        assert isinstance(analyzer, OpenAILLMProvider)
        assert isinstance(generator, ClaudeLLMProvider)
    
    @pytest.mark.asyncio
    async def test_provider_fallback(self):
        """Test fallback when primary provider fails."""
        providers = {
            "primary": MockLLMProvider(),
            "fallback": MockLLMProvider()
        }
        
        # Make primary fail
        providers["primary"].analyze = AsyncMock(side_effect=Exception("Primary failed"))
        
        # Try primary, fall back to secondary
        try:
            result = await providers["primary"].analyze("Test")
        except:
            result = await providers["fallback"].analyze("Test")
        
        # Should get mock response from fallback
        assert "patterns" in result.lower()
        data = json.loads(result)
        assert data["patterns"][0]["type"] == "workflow"