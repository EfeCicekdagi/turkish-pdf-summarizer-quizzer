# src/llm_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from transformers import pipeline
import torch

from prompts import (
    build_chunk_summarize_prompt,
    build_final_summarize_prompt,
    build_quiz_prompt,
)
from postprocess import normalize_output


@dataclass
class SummarizeResult:
    chunk_summaries: List[str]
    final_summary: str


@dataclass
class QuizResult:
    quiz_text: str


class LLMService:
    """
    Simple wrapper around a Hugging Face text2text-generation model (FLAN-T5).
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # transformers pipeline uses device index: -1 for cpu, >=0 for cuda
        device_idx = 0 if device == "cuda" else -1

        self.generator = pipeline(
            task="text2text-generation",
            model=model_name,
            device=device_idx,
        )

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ) -> str:
        out = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
        text = out[0]["generated_text"]
        return normalize_output(text)

    def summarize_chunks(self, chunks: List[str]) -> SummarizeResult:
        # 1) Map: her chunk için özet
        chunk_summaries: List[str] = []
        for i, ch in enumerate(chunks):
            prompt = build_chunk_summarize_prompt(ch)
            s = self._generate(prompt, max_new_tokens=220, temperature=0.3)
            chunk_summaries.append(s)

        # 2) Reduce: chunk özetlerini birleştir, final özet üret
        combined = "\n\n".join([f"[Özet Parça {i+1}]\n{cs}" for i, cs in enumerate(chunk_summaries)])
        final_prompt = build_final_summarize_prompt(combined)
        final_summary = self._generate(final_prompt, max_new_tokens=320, temperature=0.2)

        return SummarizeResult(chunk_summaries=chunk_summaries, final_summary=final_summary)

    def generate_quiz(self, final_summary: str, n_questions: int = 5) -> QuizResult:
        prompt = build_quiz_prompt(final_summary, n_questions=n_questions)
        quiz_text = self._generate(prompt, max_new_tokens=420, temperature=0.4)
        return QuizResult(quiz_text=quiz_text)