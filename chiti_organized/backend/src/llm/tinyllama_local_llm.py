"""
TinyLlama Local LLM Implementation for Resource-Constrained Environments
"""

from typing import Dict, Any, List, Optional, Generator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import logging
import os
from threading import Thread

logger = logging.getLogger(__name__)


class TinyLlamaLocalLLM:
    """TinyLlama 1.1B LLM for low-resource local/offline mode"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize TinyLlama model
        
        Args:
            config: Dictionary containing TinyLlama configuration
        """
        self.model_name = config["model_name"]
        self.device = config["device"]
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
        self.cache_dir = config["cache_dir"]
        
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"🤖 Initializing TinyLlama LLM: {self.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the TinyLlama model and tokenizer"""
        try:
            logger.info("📥 Loading TinyLlama model... (smaller model, faster loading)")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings for smaller model
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "low_cpu_mem_usage": True
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to specific device if not using device_map
            if self.device != "cuda" or model_kwargs.get("device_map") is None:
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            self.model_loaded = True
            
            logger.info(f"✅ TinyLlama model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Error loading TinyLlama model: {e}")
            self.model_loaded = False
            raise
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for TinyLlama chat template
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        formatted_messages = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_messages.append(f"<|system|>\n{content}</s>")
            elif role == "user":
                formatted_messages.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                formatted_messages.append(f"<|assistant|>\n{content}</s>")
        
        # Add assistant start token for generation
        formatted_messages.append("<|assistant|>")
        
        return "\n".join(formatted_messages)
    
    def generate(self, prompt: str, system_message: str = None, **kwargs) -> str:
        """Generate response using TinyLlama
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Cannot generate response.")
        
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            # Format messages for TinyLlama
            formatted_prompt = self._format_messages(messages)
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024  # Smaller context for TinyLlama
            ).to(self.device)
            
            # Generation parameters optimized for TinyLlama
            generation_params = {
                "max_new_tokens": kwargs.get("max_tokens", min(self.max_tokens, 512)),  # Conservative for small model
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 50),
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
            }
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **generation_params
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up the response
            response = response.strip()
            
            # Remove any remaining special tokens or artifacts
            stop_tokens = ["</s>", "<|system|>", "<|user|>", "<|assistant|>"]
            for stop_token in stop_tokens:
                if stop_token in response:
                    response = response.split(stop_token)[0].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            raise
    
    def generate_stream(self, prompt: str, system_message: str = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using TinyLlama
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of generated text
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Cannot generate response.")
        
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            # Format messages for TinyLlama
            formatted_prompt = self._format_messages(messages)
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Setup streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                timeout=10.0,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Generation parameters
            generation_params = {
                "max_new_tokens": kwargs.get("max_tokens", min(self.max_tokens, 512)),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 50),
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                "streamer": streamer
            }
            
            # Start generation in separate thread
            generation_thread = Thread(
                target=self.model.generate,
                args=(inputs.input_ids,),
                kwargs=generation_params
            )
            generation_thread.start()
            
            # Yield tokens as they're generated
            for token in streamer:
                if token:
                    yield token
                    
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
        # Limit context for smaller model
        context_text = "\n\n".join([f"Context {i+1}: {doc[:200]}..." if len(doc) > 200 else f"Context {i+1}: {doc}" 
                                   for i, doc in enumerate(context[:3])])  # Limit to 3 contexts
        
        if not system_message:
            system_message = """You are a helpful AI assistant. Answer the user's question based only on the provided context.
Be concise and accurate. If the context doesn't contain relevant information, say so clearly."""
        
        prompt = f"""Context:
{context_text}

Question: {query}

Answer based on the context above:"""
        
        return self.generate(prompt, system_message)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback estimation
            return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information
        
        Returns:
            Dictionary with model information
        """
        return {
            "provider": "TinyLlama",
            "model": self.model_name,
            "device": self.device,
            "parameters": "1.1B",
            "size": "~1GB",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "cache_dir": self.cache_dir,
            "type": "local",
            "loaded": self.model_loaded,
            "optimized_for": "resource-constrained environments"
        }
    
    def is_available(self) -> bool:
        """Check if the model is loaded and available
        
        Returns:
            True if model is available, False otherwise
        """
        return self.model_loaded and self.model is not None
    
    def unload_model(self):
        """Unload the model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.model_loaded = False
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🗑️  TinyLlama model unloaded from memory")