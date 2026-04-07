from typing import List
from openai import OpenAI


class OpenRouterEmbeddingFunction:
    """Custom embedding function using OpenRouter's API"""

    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = model

    def name(self) -> str:
        return self.model

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = []
        for text in input:
            try:
                response = self.client.embeddings.create(model=self.model, input=text)
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Error generating embedding: {e}")
                # Return a zero vector as fallback (adjust dimension based on model)
                embeddings.append([0.0] * 1024)  # Adjust size as needed
        return embeddings
