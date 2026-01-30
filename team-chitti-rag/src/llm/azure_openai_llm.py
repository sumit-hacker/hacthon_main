"""
Azure OpenAI LLM Implementation for Online Mode
"""

from typing import Dict, Any, List, Optional, Generator
from openai import AzureOpenAI
import logging

logger = logging.getLogger(__name__)


class AzureOpenAILLM:
    """Azure OpenAI LLM for online mode"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Azure OpenAI client
        
        Args:
            config: Dictionary containing Azure OpenAI configuration
        """
        self.endpoint = config["endpoint"]
        self.api_key = config["api_key"]
        self.api_version = config["api_version"]
        self.deployment_name = config["deployment_name"]
        self.model = config["model"]
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        logger.info(f"🌐 Initialized Azure OpenAI LLM: {self.model}")
    
    def generate(self, prompt: str, system_message: str = None, **kwargs) -> str:
        """Generate response using Azure OpenAI
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            # Override defaults with provided kwargs
            generation_params = {
                "model": self.deployment_name,  # Use deployment name for Azure
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", 0.95),
                "frequency_penalty": kwargs.get("frequency_penalty", 0),
                "presence_penalty": kwargs.get("presence_penalty", 0),
            }
            
            response = self.client.chat.completions.create(**generation_params)
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise ValueError("No response generated")
                
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            raise
    
    def generate_stream(self, prompt: str, system_message: str = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using Azure OpenAI
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of generated text
        """
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            # Override defaults with provided kwargs
            generation_params = {
                "model": self.deployment_name,  # Use deployment name for Azure
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", 0.95),
                "frequency_penalty": kwargs.get("frequency_penalty", 0),
                "presence_penalty": kwargs.get("presence_penalty", 0),
                "stream": True
            }
            
            response = self.client.chat.completions.create(**generation_params)
            
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                        
        except Exception as e:
            logger.error(f"❌ Error generating streaming response: {e}")
            raise
    
    def generate_with_context(self, query: str, context: List[str], system_message: str = None) -> str:
        """Generate response with provided context documents
        
        Args:
            query: User query
            context: List of context documents
            system_message: Optional system message
            
        Returns:
            Generated response with context
        """
        context_text = "\n\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(context)])
        
        if not system_message:
            system_message = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context doesn't contain relevant information, say so clearly."""
        
        prompt = f"""Context:
{context_text}

Question: {query}

Please provide a detailed answer based on the context above."""
        
        return self.generate(prompt, system_message)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information
        
        Returns:
            Dictionary with model information
        """
        return {
            "provider": "Azure OpenAI",
            "model": self.model,
            "deployment": self.deployment_name,
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "type": "online"
        }
    
    def is_available(self) -> bool:
        """Check if the model is available and accessible
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            # Test with a simple query
            test_response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"❌ Azure OpenAI model not available: {e}")
            return False