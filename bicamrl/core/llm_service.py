"""Centralized LLM service for all AI-powered inference in Bicamrl.

This module provides a unified interface for LLM operations throughout the system,
ensuring consistent error handling, rate limiting, and response formatting.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

from ..rate_limiter import RateLimiter
from ..sleep.llm_providers import create_llm_providers
from ..utils.logging_config import get_logger

logger = get_logger("llm_service")

T = TypeVar("T")


@dataclass
class LLMRequest:
    """Structured LLM request with metadata."""

    prompt: str
    system_prompt: Optional[str] = None
    response_format: Optional[str] = None  # 'json', 'text', 'structured'
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse(Generic[T]):
    """Structured LLM response with metadata."""

    content: T
    raw_response: str
    request_id: str
    duration_ms: float
    tokens_used: Optional[int] = None
    provider: Optional[str] = None
    error: Optional[str] = None


class LLMResponseParser(ABC):
    """Base class for parsing LLM responses into structured data."""

    @abstractmethod
    def parse(self, response: str) -> Any:
        """Parse raw LLM response into structured format."""
        pass

    @abstractmethod
    def validate(self, parsed_data: Any) -> bool:
        """Validate parsed data meets expected schema."""
        pass


class WorldModelParser(LLMResponseParser):
    """Parser for world model inference responses."""

    def parse(self, response: str) -> Dict[str, Any]:
        """Parse world model response."""
        try:
            # First, try to extract JSON from the response
            # Some models might include thinking or explanation before/after JSON
            response = response.strip()

            # Look for JSON object boundaries
            start_idx = response.find("{")
            end_idx = response.rfind("}")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx : end_idx + 1]
                data = json.loads(json_str)
            else:
                # Try parsing the whole response
                data = json.loads(response)

            # Ensure required fields exist
            required_fields = ["domain", "confidence", "entities", "relations", "goals"]
            for field in required_fields:
                if field not in data:
                    data[field] = [] if field in ["entities", "relations", "goals"] else "unknown"

            return data
        except json.JSONDecodeError as e:
            # Fall back to structured parsing
            logger.warning(
                f"Failed to parse world model response as JSON: {e}, using fallback parser"
            )
            return self._fallback_parse(response)

    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Simple fallback parser for non-JSON responses."""
        # Log the raw response for debugging
        logger.debug(f"Fallback parsing raw response: {response[:500]}...")

        # Return a basic structure with the raw response
        # Let the LLM's natural language be preserved for analysis
        result = {
            "domain": "unstructured_response",
            "confidence": 0.0,  # Zero confidence since we couldn't parse
            "entities": [],
            "relations": [],
            "goals": [],
            "raw_analysis": response,
            "parse_failed": True,  # Flag to indicate fallback was used
        }

        # If the response is very short, it might be an error
        if len(response.strip()) < 10:
            result["domain"] = "error_response"

        return result

    def validate(self, parsed_data: Any) -> bool:
        """Validate world model data structure."""
        if not isinstance(parsed_data, dict):
            return False

        required = ["domain", "entities", "relations", "goals"]
        return all(key in parsed_data for key in required)


class LLMService:
    """Centralized service for all LLM operations in Bicamrl.

    This service provides:
    - Unified interface for different LLM providers
    - Automatic retries and error handling
    - Rate limiting and quota management
    - Response parsing and validation
    - Metrics and logging
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Handle empty providers config for mock
        llm_config = config.get("llm_providers", {})
        if not llm_config and config.get("default_provider") == "mock":
            # Use mock provider configuration
            llm_config = {"mock": {}}
        self.providers = create_llm_providers(llm_config)
        self.default_provider = config.get("default_provider")

        if not self.default_provider:
            raise ValueError(
                "No default_provider specified in configuration. "
                "Please set 'default_provider' in your Mind.toml file."
            )

        if self.default_provider not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(
                f"Default provider '{self.default_provider}' not found in configured providers. "
                f"Available providers: {available}. "
                f"Please check your Mind.toml configuration."
            )
        self.rate_limiter = RateLimiter(requests_per_minute=config.get("rate_limit", 60))
        self.parsers = {
            "world_model": WorldModelParser(),
        }
        self._request_count = 0

    async def infer_world_model(self, interaction: Dict[str, Any]) -> LLMResponse[Dict[str, Any]]:
        """Infer world model from interaction using LLM.

        This replaces the hardcoded pattern matching with dynamic LLM inference.
        """
        prompt = self._build_world_model_prompt(interaction)

        request = LLMRequest(
            prompt=prompt,
            system_prompt="""You are an expert at understanding user goals and building world models.
            Analyze the interaction and infer:
            1. The domain (can be anything - coding, writing, research, music, cooking, etc.)
            2. Key entities involved (files, concepts, systems, data, etc.)
            3. Relationships between entities
            4. The user's likely goals

            IMPORTANT: You MUST respond with ONLY valid JSON, no other text before or after.
            Start your response with { and end with }

            Example response format:
            {
                "domain": "cooking",
                "confidence": 0.8,
                "entities": [
                    {"id": "chocolate_cake.md", "type": "recipe_file", "properties": {"category": "dessert"}}
                ],
                "relations": [
                    {"source": "user", "target": "chocolate_cake.md", "type": "creates"}
                ],
                "goals": [
                    {"type": "recipe_creation", "description": "Create a chocolate cake recipe", "confidence": 0.9}
                ]
            }

            Your response should follow this exact structure. Be creative in identifying domains.
            Entity and relation types should be descriptive and domain-appropriate.

            Remember: Output ONLY the JSON object, nothing else.""",
            response_format="json",
            temperature=0.3,  # Lower temperature for more consistent inference
        )

        response = await self._execute_request(request, parser="world_model")
        return response

    async def enhance_prompt(
        self, original_prompt: str, context: Dict[str, Any]
    ) -> LLMResponse[str]:
        """Enhance a prompt based on learned patterns and context."""
        request = LLMRequest(
            prompt=f"Original prompt: {original_prompt}\n\nContext: {json.dumps(context, indent=2)}",
            system_prompt="You are an expert at improving prompts based on context and patterns. Enhance the prompt to be clearer and more effective.",
            temperature=0.7,
        )

        return await self._execute_request(request)

    async def analyze_patterns(
        self, interactions: List[Dict[str, Any]]
    ) -> LLMResponse[Dict[str, Any]]:
        """Analyze interactions to discover patterns."""
        request = LLMRequest(
            prompt=self._build_pattern_analysis_prompt(interactions),
            system_prompt="You are an expert at discovering patterns in user behavior. Identify recurring workflows, preferences, and habits.",
            response_format="json",
            temperature=0.5,
        )

        return await self._execute_request(request)

    async def _execute_request(
        self, request: LLMRequest, provider: Optional[str] = None, parser: Optional[str] = None
    ) -> LLMResponse:
        """Execute an LLM request with error handling and parsing."""
        start_time = datetime.now()
        request_id = f"req_{self._request_count}_{start_time.timestamp()}"
        self._request_count += 1

        # Rate limiting - use a default client ID for now
        allowed, retry_after = await self.rate_limiter.check_rate_limit("llm_service", 1.0)
        if not allowed:
            logger.warning(f"Rate limit exceeded, retry after {retry_after} seconds")
            # For now, just wait instead of failing
            if retry_after:
                await asyncio.sleep(retry_after)

        provider_name = provider or self.default_provider
        llm_provider = self.providers.get(provider_name)

        if not llm_provider:
            logger.error(f"Provider {provider_name} not found")
            return LLMResponse(
                content=None,
                raw_response="",
                request_id=request_id,
                duration_ms=0,
                error=f"Provider {provider_name} not configured",
            )

        try:
            # Build the full prompt
            full_prompt = request.prompt

            # For LM Studio and local models, avoid analyze() with JSON format
            # as many local models don't support structured output
            is_local_model = provider_name in ["lmstudio", "local", "mock"]

            # Make the request
            if request.system_prompt and not is_local_model:
                raw_response = await llm_provider.analyze(
                    full_prompt, system=request.system_prompt, temperature=request.temperature
                )
            else:
                # For local models or when no system prompt, use generate
                if request.system_prompt:
                    # Include system prompt in the main prompt for local models
                    full_prompt = f"{request.system_prompt}\n\n{full_prompt}"
                raw_response = await llm_provider.generate(
                    full_prompt, temperature=request.temperature
                )

            # Parse response if parser specified
            if parser and parser in self.parsers:
                parsed_content = self.parsers[parser].parse(raw_response)
                if not self.parsers[parser].validate(parsed_content):
                    logger.warning(f"Parsed response failed validation for parser {parser}")
            else:
                parsed_content = raw_response

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            logger.info(
                "LLM request completed",
                extra={
                    "request_id": request_id,
                    "provider": provider_name,
                    "duration_ms": duration_ms,
                    "response_format": request.response_format,
                },
            )

            return LLMResponse(
                content=parsed_content,
                raw_response=raw_response,
                request_id=request_id,
                duration_ms=duration_ms,
                provider=provider_name,
            )

        except Exception as e:
            logger.error(
                "LLM request failed",
                extra={"request_id": request_id, "provider": provider_name, "error": str(e)},
                exc_info=True,
            )

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            return LLMResponse(
                content=None,
                raw_response="",
                request_id=request_id,
                duration_ms=duration_ms,
                provider=provider_name,
                error=str(e),
            )

    def _build_world_model_prompt(self, interaction: Dict[str, Any]) -> str:
        """Build prompt for world model inference."""
        # Extract relevant parts
        user_query = interaction.get("user_query", "")
        actions = interaction.get("actions_taken", [])

        prompt = f"""Analyze this interaction and build a world model.

INTERACTION DATA:
User Query: {user_query}
Actions Taken: {json.dumps(actions, indent=2)}
Success: {interaction.get("success", False)}
Feedback: {interaction.get("feedback_type", "none")}

TASK: Based on this interaction, create a JSON object that captures:
1. The domain the user is working in (be specific and creative)
2. Key entities involved (with meaningful IDs and types)
3. Relationships between entities
4. The user's goals (what they're trying to achieve)

IMPORTANT INSTRUCTIONS:
- Output ONLY a JSON object, no explanations or other text
- Start with {{ and end with }}
- Use the exact structure shown in the system prompt
- Be creative with domains - don't default to generic categories
- Make entity IDs match actual items from the interaction
- Infer meaningful relationships based on the actions taken

Begin your JSON response now:"""

        return prompt

    def _build_pattern_analysis_prompt(self, interactions: List[Dict[str, Any]]) -> str:
        """Build prompt for pattern analysis."""
        # Summarize interactions
        summaries = []
        for i, interaction in enumerate(interactions[-10:]):  # Last 10 interactions
            summaries.append(
                {
                    "query": interaction.get("user_query", "")[:100],
                    "actions": len(interaction.get("actions_taken", [])),
                    "success": interaction.get("success", False),
                }
            )

        prompt = f"""Analyze these recent interactions to identify patterns:

{json.dumps(summaries, indent=2)}

Identify:
1. Recurring workflows or sequences
2. Common goals or objectives
3. Preferred tools or approaches
4. Areas where the user might benefit from automation

Respond with specific, actionable patterns in JSON format."""

        return prompt

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        provider: Optional[str] = None,
    ) -> str:
        """Generate a response from the LLM.

        This is a simplified interface for Wake agent and other components
        that need basic text generation without structured parsing.
        """
        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            response_format="text",
        )

        response = await self._execute_request(request, provider=provider)

        # Return the content directly (for text responses, this is the raw string)
        if response.error:
            raise Exception(f"LLM generation failed: {response.error}")

        return response.content if response.content else response.raw_response
