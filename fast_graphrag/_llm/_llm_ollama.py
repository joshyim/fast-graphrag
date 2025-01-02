"""LLM Services module for Ollama."""
import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Type

import aiohttp
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fast_graphrag._exceptions import LLMServiceNoResponseError
from fast_graphrag._types import BaseModelAlias
from fast_graphrag._utils import logger, throttle_async_func_call
from ._base import BaseEmbeddingService, BaseLLMService, T_model

TIMEOUT_SECONDS = 180.0
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

@dataclass
class OllamaLLMService(BaseLLMService):
    """LLM Service for Ollama local models."""

    model: Optional[str] = field(default="llama2")
    base_url: str = field(default=DEFAULT_OLLAMA_HOST)
    
    def __post_init__(self):
        """Initialize the Ollama service."""
        self.base_url = self.base_url.rstrip('/')
        logger.debug(f"Initialized OllamaLLMService with base URL: {self.base_url}")

    @throttle_async_func_call(max_concurrent=int(os.getenv("CONCURRENT_TASK_LIMIT", 1024)), stagger_time=0.001, waiting_time=0.001)
    async def send_message(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        response_model: Type[T_model] | None = None,
        **kwargs: Any,
    ) -> Tuple[T_model, list[dict[str, str]]]:
        """Send a message to the Ollama model and receive a response.

        Args:
            prompt (str): The input message to send to the language model.
            model (str): The name of the model to use. Defaults to the model provided in the config.
            system_prompt (str, optional): The system prompt to set the context for the conversation.
            history_messages (list, optional): A list of previous messages in the conversation.
            response_model (Type[T], optional): The Pydantic model to parse the response.
            **kwargs: Additional keyword arguments for the model.

        Returns:
            Tuple[T_model, list[dict[str, str]]]: The parsed response and updated message history.
        """
        model = model or self.model
        if model is None:
            raise ValueError("Model name must be provided.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history_messages:
            messages.extend(history_messages)
        
        messages.append({"role": "user", "content": prompt})

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            stop=stop_after_attempt(3),
        ):
            with attempt:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": model,
                            "messages": messages,
                            "stream": False,
                            **kwargs
                        },
                        timeout=TIMEOUT_SECONDS
                    ) as response:
                        if response.status != 200:
                            raise LLMServiceNoResponseError(
                                f"Ollama API returned status code {response.status}"
                            )
                        
                        result = await response.json()
                        
                        if response_model:
                            try:
                                parsed_response = response_model.model_validate_json(
                                    result["message"]["content"]
                                )
                            except Exception as e:
                                logger.error(f"Failed to parse response: {e}")
                                raise LLMServiceNoResponseError(
                                    f"Failed to parse response: {e}"
                                )
                        else:
                            parsed_response = result["message"]["content"]

                        messages.append({
                            "role": "assistant",
                            "content": result["message"]["content"]
                        })
                        
                        return parsed_response, messages


@dataclass
class OllamaEmbeddingService(BaseEmbeddingService):
    """Embedding Service for Ollama local models."""

    embedding_dim: int = field(default=4096)  # Default dimension for Llama2 embeddings
    max_elements_per_request: int = field(default=32)
    model: Optional[str] = field(default="llama2")
    base_url: str = field(default=DEFAULT_OLLAMA_HOST)

    def __post_init__(self):
        """Initialize the Ollama embedding service."""
        self.base_url = self.base_url.rstrip('/')
        logger.debug(f"Initialized OllamaEmbeddingService with base URL: {self.base_url}")

    async def encode(
        self, texts: list[str], model: Optional[str] = None
    ) -> List[List[float]]:
        """Get the embedding representation of the input texts.

        Args:
            texts (list[str]): The input texts to embed.
            model (str, optional): The name of the model to use.

        Returns:
            List[List[float]]: The embedding vectors as a list of float lists.
        """
        model = model or self.model
        if model is None:
            raise ValueError("Model name must be provided.")

        embeddings = []
        for i in range(0, len(texts), self.max_elements_per_request):
            batch = texts[i:i + self.max_elements_per_request]
            batch_embeddings = await self._embedding_request(batch, model)
            embeddings.extend(batch_embeddings)
        
        return embeddings

    async def _embedding_request(self, texts: List[str], model: str) -> List[List[float]]:
        """Make an embedding request to the Ollama API.

        Args:
            texts (List[str]): Batch of texts to embed.
            model (str): Model name to use.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        embeddings = []
        
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            stop=stop_after_attempt(3),
        ):
            with attempt:
                async with aiohttp.ClientSession() as session:
                    for text in texts:
                        async with session.post(
                            f"{self.base_url}/api/embeddings",
                            json={"model": model, "prompt": text},
                            timeout=TIMEOUT_SECONDS
                        ) as response:
                            if response.status != 200:
                                raise LLMServiceNoResponseError(
                                    f"Ollama API returned status code {response.status}"
                                )
                            
                            result = await response.json()
                            embeddings.append(result["embedding"])

        return embeddings