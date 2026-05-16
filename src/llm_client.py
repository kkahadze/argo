#!/usr/bin/env python3
"""
LLM Client abstraction layer to support multiple LLM providers (OpenAI, Anthropic Claude, Google Gemini, etc.)
"""
import os
from typing import Optional, Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Provider type
LLMProvider = Literal["openai", "anthropic", "gemini"]
DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-5.4-nano",
    "anthropic": "claude-sonnet-4-5-20250929",
    "gemini": "gemini-3.1-flash-lite-preview",
}

class LLMClient:
    """
    Unified interface for different LLM providers.
    Supports OpenAI (GPT), Anthropic (Claude), and Google (Gemini).
    """
    
    def __init__(
        self, 
        provider: LLMProvider = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: Which LLM provider to use ("openai", "anthropic", or "gemini")
            api_key: API key for the provider (if None, will try to get from env)
            model: Model name to use (if None, will use default for provider)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response (optional)
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set up provider-specific configuration
        if provider == "openai":
            import openai
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            # Use model from parameter, then env var LLM_MODEL, then default
            self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL_BY_PROVIDER["openai"]
            
            if not self.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env or pass api_key parameter.")
            
        elif provider == "anthropic":
            try:
                import anthropic
                self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
                # Use model from parameter, then env var LLM_MODEL, then default
                self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL_BY_PROVIDER["anthropic"]
                
                if not self.api_key:
                    raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY in .env or pass api_key parameter.")
                
                self.client = anthropic.Anthropic(api_key=self.api_key)
                
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install it with: pip install anthropic"
                )
        
        elif provider == "gemini":
            try:
                from google import genai
                self.api_key = api_key or os.getenv("GEMINI_API_KEY")
                # Use model from parameter, then env var LLM_MODEL, then default
                self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL_BY_PROVIDER["gemini"]
                
                if not self.api_key:
                    raise ValueError("Gemini API key not found. Set GEMINI_API_KEY in .env or pass api_key parameter.")
                
                # Initialize the new genai client
                self.genai_client = genai.Client(api_key=self.api_key)
                
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. Install it with: pip install google-genai"
                )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', or 'gemini'.")
    
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a prompt to the LLM and get a completion.
        
        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt (for providers that support it)
            
        Returns:
            str: The LLM's response text
        """
        if self.provider == "openai":
            return self._complete_openai(prompt, system_prompt)
        elif self.provider == "anthropic":
            return self._complete_anthropic(prompt, system_prompt)
        elif self.provider == "gemini":
            return self._complete_gemini(prompt, system_prompt)
    
    def _complete_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Complete using OpenAI API."""
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        
        # GPT-5 family models use the v1/responses endpoint.
        is_gpt5_family = self.model.lower().startswith("gpt-5")

        if is_gpt5_family:
            # Use the v1/responses endpoint for GPT-5 family models
            # GPT-5 Pro uses 'input' parameter (not 'prompt')
            # Input can be either a string or a messages array
            if system_prompt:
                # Use messages array format when we have a system prompt
                input_data = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                # Use messages array format for consistency
                input_data = [
                    {"role": "user", "content": prompt}
                ]
            
            kwargs = {
                "model": self.model,
                "input": input_data,
            }
            
            # Add optional parameters
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.max_tokens:
                kwargs["max_output_tokens"] = self.max_tokens  # Note: different param name
            
            response = client.responses.create(**kwargs)
            
            # Extract text from response
            return response.output_text
        else:
            # Use standard chat completions for other models
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
    
    def _complete_anthropic(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Complete using Anthropic Claude API."""
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens or 4096,  # Claude requires max_tokens
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**kwargs)
        return response.content[0].text
    
    def _complete_gemini(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Complete using Google Gemini API with the new google.genai SDK."""
        # Build the configuration
        config = {
            "temperature": self.temperature,
        }
        
        if self.max_tokens:
            config["max_output_tokens"] = self.max_tokens
        
        # Build contents list
        contents = []
        if system_prompt:
            contents.append({"role": "system", "parts": [{"text": system_prompt}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        # Generate response using new SDK
        response = self.genai_client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        
        return response.text


def get_default_llm_client(
    provider: Optional[LLMProvider] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> LLMClient:
    """
    Get a default LLM client instance.
    
    Args:
        provider: Provider to use (if None, checks LLM_PROVIDER env var, defaults to "openai")
        model: Model to use (if None, checks LLM_MODEL env var, then uses provider's default)
        api_key: API key (if None, uses environment variables)
        
    Returns:
        LLMClient: Configured LLM client
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai")
    
    if model is None:
        model = os.getenv("LLM_MODEL")
    
    return LLMClient(provider=provider, model=model, api_key=api_key)


# Convenience functions for backward compatibility
def complete_with_openai(prompt: str, model: str = "gpt-5.4-nano", api_key: Optional[str] = None) -> str:
    """
    Quick function to complete a prompt with OpenAI.
    
    Args:
        prompt: The prompt text
        model: OpenAI model name
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        str: The response text
    """
    client = LLMClient(provider="openai", model=model, api_key=api_key)
    return client.complete(prompt)


def complete_with_claude(prompt: str, model: str = "claude-sonnet-4-5-20250929", api_key: Optional[str] = None) -> str:
    """
    Quick function to complete a prompt with Claude.
    
    Args:
        prompt: The prompt text
        model: Claude model name
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        str: The response text
    """
    client = LLMClient(provider="anthropic", model=model, api_key=api_key)
    return client.complete(prompt)


def complete_with_gemini(
    prompt: str,
    model: str = "gemini-3.1-flash-lite-preview",
    api_key: Optional[str] = None,
) -> str:
    """
    Quick function to complete a prompt with Gemini.
    
    Args:
        prompt: The prompt text
        model: Gemini model name
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        str: The response text
    """
    client = LLMClient(provider="gemini", model=model, api_key=api_key)
    return client.complete(prompt)
