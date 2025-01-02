"""Test module for Ollama LLM service."""
import asyncio
import pytest
from pydantic import BaseModel

from fast_graphrag._llm._llm_ollama import OllamaLLMService, OllamaEmbeddingService


class TestResponse(BaseModel):
    """Test response model."""
    message: str
    confidence: float


async def test_ollama_chat():
    """Test basic chat completion with Ollama."""
    service = OllamaLLMService(model="llama2")
    
    # Simple chat completion
    response, history = await service.send_message(
        prompt="What is the capital of France?",
    )
    assert isinstance(response, str)
    assert len(history) == 2  # Initial prompt + response
    
    # Chat with system prompt
    response, history = await service.send_message(
        prompt="Tell me a short joke",
        system_prompt="You are a helpful assistant who tells family-friendly jokes",
    )
    assert isinstance(response, str)
    assert len(history) == 3  # System + prompt + response
    
    # Chat with structured response
    response, history = await service.send_message(
        prompt="Rate how confident you are that Paris is the capital of France",
        response_model=TestResponse,
    )
    assert isinstance(response, TestResponse)
    assert hasattr(response, 'message')
    assert hasattr(response, 'confidence')


async def test_ollama_embeddings():
    """Test text embeddings with Ollama."""
    service = OllamaEmbeddingService(model="llama2")
    
    # Single text embedding
    texts = ["This is a test sentence."]
    embeddings = await service.encode(texts)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == service.embedding_dim
    
    # Batch text embeddings
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    embeddings = await service.encode(texts)
    assert len(embeddings) == 3
    assert all(len(emb) == service.embedding_dim for emb in embeddings)


if __name__ == "__main__":
    # Example usage without pytest
    async def main():
        # Initialize the service
        chat_service = OllamaLLMService(
            model="llama2",
            base_url="http://localhost:11434"  # default Ollama endpoint
        )
        
        # Simple chat example
        print("\n=== Simple Chat Example ===")
        response, _ = await chat_service.send_message(
            prompt="What are the three laws of robotics?"
        )
        print(f"Response: {response}")
        
        # Structured response example
        print("\n=== Structured Response Example ===")
        response, _ = await chat_service.send_message(
            prompt="Rate how confident you are about the three laws of robotics",
            response_model=TestResponse
        )
        print(f"Message: {response.message}")
        print(f"Confidence: {response.confidence}")
        
        # Embedding example
        print("\n=== Embedding Example ===")
        embedding_service = OllamaEmbeddingService(model="llama2")
        embeddings = await embedding_service.encode(
            ["What are the three laws of robotics?"]
        )
        print(f"Embedding dimension: {len(embeddings[0])}")
        print(f"First few values: {embeddings[0][:5]}")

    asyncio.run(main())
