"""LLM provider implementations for Sleep Layer."""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import aiohttp

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.api_key = api_key
        self.config = config or {}
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        
    @abstractmethod
    async def analyze(self, prompt: str, **kwargs) -> str:
        """Analyze content and return insights."""
        pass
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content based on prompt."""
        pass
        
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

class ClaudeLLMProvider(BaseLLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key or os.getenv('ANTHROPIC_API_KEY'), config)
        self.base_url = "https://api.anthropic.com/v1"
        self.model = self.config.get('model', 'claude-3-opus-20240229')
        self.max_tokens = self.config.get('max_tokens', 4096)
        
    async def _make_request(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        """Make a request to Claude API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages
        }
        
        if system:
            data["system"] = system
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result['content'][0]['text']
                
    async def analyze(self, prompt: str, **kwargs) -> str:
        """Analyze content using Claude."""
        system = kwargs.get('system', "You are an AI system analyzer. Provide detailed, structured analysis.")
        messages = [{"role": "user", "content": prompt}]
        
        return await self._retry_with_backoff(self._make_request, messages, system)
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using Claude."""
        system = kwargs.get('system', "You are a helpful AI assistant.")
        messages = [{"role": "user", "content": prompt}]
        
        return await self._retry_with_backoff(self._make_request, messages, system)

class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI GPT provider (also supports OpenAI-compatible APIs like LM Studio)."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key or os.getenv('OPENAI_API_KEY'), config)
        # Support custom base URL for OpenAI-compatible APIs (e.g., LM Studio)
        self.base_url = self.config.get('base_url', os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
        self.model = self.config.get('model', 'gpt-4-turbo-preview')
        self.max_tokens = self.config.get('max_tokens', 4096)
        
    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make a request to OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": kwargs.get('temperature', 0.7)
        }
        
        if 'response_format' in kwargs:
            data['response_format'] = kwargs['response_format']
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result['choices'][0]['message']['content']
                
    async def analyze(self, prompt: str, **kwargs) -> str:
        """Analyze content using GPT."""
        messages = [
            {"role": "system", "content": "You are an AI system analyzer. Provide detailed, structured analysis."},
            {"role": "user", "content": prompt}
        ]
        
        # Try to get JSON response for analysis
        kwargs['response_format'] = {"type": "json_object"}
        
        return await self._retry_with_backoff(self._make_request, messages, **kwargs)
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using GPT."""
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        return await self._retry_with_backoff(self._make_request, messages, **kwargs)

class LocalLLMProvider(BaseLLMProvider):
    """Local LLM provider (e.g., Ollama, LlamaCpp)."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key, config)
        self.base_url = self.config.get('base_url', 'http://localhost:11434')
        self.model = self.config.get('model', 'llama2')
        
    async def _make_request(self, prompt: str, **kwargs) -> str:
        """Make a request to local LLM."""
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        if 'system' in kwargs:
            data['system'] = kwargs['system']
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result['response']
                
    async def analyze(self, prompt: str, **kwargs) -> str:
        """Analyze content using local LLM."""
        system = "You are an AI system analyzer. Provide detailed, structured analysis."
        full_prompt = f"{system}\n\n{prompt}"
        
        return await self._retry_with_backoff(self._make_request, full_prompt, **kwargs)
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using local LLM."""
        return await self._retry_with_backoff(self._make_request, prompt, **kwargs)

class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""
    
    async def analyze(self, prompt: str, **kwargs) -> str:
        """Mock analysis."""
        return json.dumps({
            "patterns": [
                {
                    "type": "workflow",
                    "description": "Test pattern detected",
                    "frequency": 5,
                    "recommendation": "Consider automating this workflow",
                    "confidence": 0.8
                }
            ]
        })
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Mock generation."""
        return "Mock enhanced prompt: " + prompt[:50] + "..."

class MultiLLMCoordinator:
    """Coordinates multiple LLM providers for different tasks."""
    
    def __init__(self, providers: Dict[str, BaseLLMProvider]):
        self.providers = providers
        
    def get_provider(self, role: str) -> Optional[BaseLLMProvider]:
        """Get provider for a specific role."""
        # Map roles to providers
        role_mapping = {
            'analyzer': 'analyzer',
            'enhancer': 'generator',
            'optimizer': 'analyzer',
            'synthesizer': 'generator',
            'reviewer': 'analyzer'
        }
        
        provider_key = role_mapping.get(role, role)
        return self.providers.get(provider_key)
        
    async def consensus_analysis(self, prompt: str, providers: List[str]) -> Dict[str, Any]:
        """Get consensus from multiple providers."""
        results = {}
        
        # Gather responses from all providers
        tasks = []
        for provider_name in providers:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                task = asyncio.create_task(provider.analyze(prompt))
                tasks.append((provider_name, task))
                
        # Collect results
        for provider_name, task in tasks:
            try:
                result = await task
                results[provider_name] = result
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                results[provider_name] = None
                
        # Simple consensus: majority vote or average confidence
        return self._compute_consensus(results)
        
    def _compute_consensus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute consensus from multiple results."""
        valid_results = [r for r in results.values() if r is not None]
        
        if not valid_results:
            return {"consensus": None, "confidence": 0.0}
            
        # For now, just return the first valid result
        # TODO: Implement proper consensus algorithm
        return {
            "consensus": valid_results[0],
            "confidence": len(valid_results) / len(results),
            "individual_results": results
        }

def create_llm_providers(config: Dict[str, Any]) -> Dict[str, BaseLLMProvider]:
    """Create LLM providers from configuration."""
    providers = {}
    
    # Create providers based on config
    if 'claude' in config and config['claude'].get('api_key'):
        providers['claude'] = ClaudeLLMProvider(
            api_key=config['claude']['api_key'],
            config=config['claude']
        )
        
    if 'openai' in config and config['openai'].get('api_key'):
        providers['openai'] = OpenAILLMProvider(
            api_key=config['openai']['api_key'],
            config=config['openai']
        )
    
    # Support LM Studio or other OpenAI-compatible APIs
    if 'lmstudio' in config:
        providers['lmstudio'] = OpenAILLMProvider(
            api_key=config['lmstudio'].get('api_key', 'not-needed'),
            config=config['lmstudio']
        )
        
    if 'local' in config and config['local'].get('enabled'):
        providers['local'] = LocalLLMProvider(config=config['local'])
        
    # Add mock provider for testing
    providers['mock'] = MockLLMProvider()
    
    # Set up role mappings
    role_config = config.get('roles', {})
    role_providers = {}
    
    for role, provider_name in role_config.items():
        if provider_name in providers:
            role_providers[role] = providers[provider_name]
            
    # Default mappings if not specified
    if 'analyzer' not in role_providers and 'openai' in providers:
        role_providers['analyzer'] = providers['openai']
    elif 'analyzer' not in role_providers and 'claude' in providers:
        role_providers['analyzer'] = providers['claude']
        
    if 'generator' not in role_providers and 'claude' in providers:
        role_providers['generator'] = providers['claude']
    elif 'generator' not in role_providers and 'openai' in providers:
        role_providers['generator'] = providers['openai']
        
    # Use mock for missing roles in development
    for role in ['analyzer', 'generator', 'enhancer', 'optimizer']:
        if role not in role_providers:
            role_providers[role] = providers.get('mock')
            
    return role_providers