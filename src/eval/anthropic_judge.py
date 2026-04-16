"""
anthropic_judge.py
Custom DeepEval judge model using Anthropic Claude instead of OpenAI.
"""
import os
from typing import Optional
from deepeval.models.base_model import DeepEvalBaseLLM
import anthropic


class AnthropicJudge(DeepEvalBaseLLM):
    """Wrapper to use Anthropic Claude as DeepEval LLM judge."""

    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model
