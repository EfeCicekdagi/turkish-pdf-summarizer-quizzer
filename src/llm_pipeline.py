# src/llm_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from transformers import pipeline
import torch

from src.prompts import build_quiz_prompt
from src.postprocess import normalize_output
from src.extractive import extractive_summary


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

        self.summarizer = pipeline(
            task="text2text-generation",
            model=summarizer_model,
            device=device_idx,
        )

        self.quizzer = pipeline(
            task="text2text-generation",
            model=quiz_model,
            device=device_idx,
        )

        self._sum_tokenizer = self.summarizer.tokenizer
        self._sum_model = self.summarizer.model

        self._quiz_tokenizer = self.quizzer.tokenizer
        self._quiz_model = self.quizzer.model

    # ✅ BURASI __init__ DIŞINDA, class seviyesinde olmalı
    @staticmethod
    def _safe_model_max_length(tokenizer, model) -> int:
        candidates = []
        cfg = getattr(model, "config", None)

        for attr in ("max_position_embeddings", "n_positions", "max_length"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0:
                candidates.append(v)

        tmax = getattr(tokenizer, "model_max_length", None)
        if isinstance(tmax, int) and tmax > 0:
            candidates.append(tmax)

        max_len = min(candidates) if candidates else 512

        # dev placeholder'ı clamp et
        if max_len > 1_000_000:
            max_len = 512

        return int(max_len)


    def _truncate_to_model_limit(self, tokenizer, model, text: str, reserve_new_tokens: int) -> str:
        """
        Truncate INPUT to fit model max length, reserving room for output tokens.
        """
        model_max = self._safe_model_max_length(tokenizer, model)

        # Leave room for generated tokens + small buffer
        max_input_tokens = max(32, model_max - int(reserve_new_tokens) - 8)

        enc = tokenizer(
            text,
            truncation=True,
            max_length=int(max_input_tokens),
            return_tensors=None,
        )

        return tokenizer.decode(enc["input_ids"], skip_special_tokens=True)

    def _generate(
        self,
        gen_pipe,
        tokenizer,
        model,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        deterministic: bool = True,
        anti_repeat: bool = True,
    ) -> str:
        # Truncate input safely (token-based)
        prompt = self._truncate_to_model_limit(tokenizer, model, prompt, reserve_new_tokens=max_new_tokens)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
        }

        # Anti-hallucination / anti-garbage defaults
        if anti_repeat:
            gen_kwargs.update(
                {
                    "no_repeat_ngram_size": 3,
                    "repetition_penalty": 1.08,
                    "length_penalty": 1.0,
                    "early_stopping": True,
                }
            )

        if deterministic:
            gen_kwargs.update(
                {
                    "do_sample": False,
                    "num_beams": 4,
                }
            )
        else:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": 0.9,
                }
            )

        out = gen_pipe(prompt, **gen_kwargs)
        text = out[0]["generated_text"]
        return normalize_output(text)

    # -----------------------------
    # Summarization
    # -----------------------------
    def summarize_chunks(
        self,
        chunks: List[str],
        mode: str = "extractive",
        n_sentences_per_chunk: int = 5,
    ) -> SummarizeResult:
        """
        Summarize each chunk, then combine.

        mode:
          "extractive"  — TF-IDF sentence extraction (faithful, zero hallucination)
          "abstractive" — mT5 generation (fluent but may hallucinate)
        """
        chunk_summaries: List[str] = []

        if mode == "extractive":
            # MAP: pick most important sentences from each chunk
            for ch in chunks:
                s = extractive_summary(ch, n_sentences=n_sentences_per_chunk)
                chunk_summaries.append(s)
        else:
            # MAP: abstractive summarization with mT5
            for ch in chunks:
                s = self._generate(
                    gen_pipe=self.summarizer,
                    tokenizer=self._sum_tokenizer,
                    model=self._sum_model,
                    prompt=ch,
                    max_new_tokens=220,
                    temperature=0.0,
                    deterministic=True,
                    anti_repeat=True,
                )
                chunk_summaries.append(s)

        # REDUCE: combine all chunk summaries with section labels
        final_summary = self._combine_summaries(chunk_summaries)

        return SummarizeResult(chunk_summaries=chunk_summaries, final_summary=final_summary)

    def _combine_summaries(self, summaries: List[str]) -> str:
        """
        mT5-base has only 512 tokens max. With ~260 reserved for generation,
        barely ONE chunk summary fits in the input window — everything after
        chunk 1 is silently truncated by _truncate_to_model_limit.

        Root fix: do NOT re-feed combined summaries into mT5 for REDUCE.
        Instead, build the final summary by concatenating all chunk summaries
        with clear section headers. This is accurate, fast, and scalable.
        """
        if not summaries:
            return ""
        if len(summaries) == 1:
            return summaries[0]

        parts = []
        for idx, s in enumerate(summaries, start=1):
            parts.append(f"── Bölüm {idx} ──\n{s.strip()}")

        return "\n\n".join(parts)

    # -----------------------------
    # Quiz generation
    # -----------------------------
    def generate_quiz(self, final_summary: str, n_questions: int = 5) -> QuizResult:
        prompt = build_quiz_prompt(final_summary, n_questions=n_questions)

        quiz_text = self._generate(
            gen_pipe=self.quizzer,
            tokenizer=self._quiz_tokenizer,
            model=self._quiz_model,
            prompt=prompt,
            max_new_tokens=420,
            temperature=0.4,
            deterministic=False,   # quizde çeşitlilik iyi
            anti_repeat=True,
        )
        return QuizResult(quiz_text=quiz_text)