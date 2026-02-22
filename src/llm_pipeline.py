# src/llm_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from transformers import pipeline
import torch

from src.prompts import (
    build_chunk_summarize_prompt,
    build_final_summarize_prompt,
    build_quiz_prompt,
)
from src.postprocess import normalize_output


@dataclass
class SummarizeResult:
    chunk_summaries: List[str]
    final_summary: str


@dataclass
class QuizResult:
    quiz_text: str


class LLMService:
    """
    Two-model setup:
    - summarizer: Turkish summarization fine-tuned model (mT5)
    - quizzer: instruction-ish model for quiz generation (FLAN-T5)
    """

    def __init__(
        self,
        summarizer_model: str = "mukayese/mt5-base-turkish-summarization",
        quiz_model: str = "google/flan-t5-base",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device_idx = 0 if device == "cuda" else -1

        # Summarizer pipeline (text2text-generation)
        self.summarizer = pipeline(
            task="text2text-generation",
            model=summarizer_model,
            device=device_idx,
        )

        # Quiz generator pipeline (text2text-generation)
        self.quizzer = pipeline(
            task="text2text-generation",
            model=quiz_model,
            device=device_idx,
        )

    def _generate(
        self,
        gen_pipe,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        deterministic: bool = True,
    ) -> str:
        # Deterministic generation reduces hallucinations & randomness
        if deterministic:
            out = gen_pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
            )
        else:
            out = gen_pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )

        text = out[0]["generated_text"]
        return normalize_output(text)

    def summarize_chunks(self, chunks: List[str]) -> SummarizeResult:
        chunk_summaries: List[str] = []

        # MAP: each chunk summarized with Turkish summarizer
        for ch in chunks:
            prompt = build_chunk_summarize_prompt(ch)
            s = self._generate(
                self.summarizer,
                prompt,
                max_new_tokens=220,
                temperature=0.3,
                deterministic=True,
            )
            chunk_summaries.append(s)

        # REDUCE: combine summaries and make structured final summary
        combined = "\n\n".join([f"[Özet Parça {i+1}]\n{cs}" for i, cs in enumerate(chunk_summaries)])
        final_prompt = build_final_summarize_prompt(combined)
        final_summary = self._generate(
            self.summarizer,
            final_prompt,
            max_new_tokens=320,
            temperature=0.2,
            deterministic=True,
        )

        return SummarizeResult(chunk_summaries=chunk_summaries, final_summary=final_summary)

    def generate_quiz(self, final_summary: str, n_questions: int = 5) -> QuizResult:
        prompt = build_quiz_prompt(final_summary, n_questions=n_questions)
        quiz_text = self._generate(
            self.quizzer,
            prompt,
            max_new_tokens=420,
            temperature=0.4,
            deterministic=False,  # quizde biraz çeşitlilik iyi
        )
        return QuizResult(quiz_text=quiz_text)