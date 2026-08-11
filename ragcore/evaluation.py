"""LLM-as-judge evaluation -- closing the retrieval->answer feedback loop.

After the main answer is generated we (optionally, on a sampled basis) ask a
cheaper model to score two things against a rubric:

* **faithfulness**    -- did the answer stay within what the context says?
* **answer_relevance**-- did the answer actually address the question?

Scores are logged on the trace so you can query "all requests where
faithfulness < 0.7 in the last 7 days" and drill into the chunk-level attribution
to classify the failure (wrong doc / wrong section / model ignored context).
"""

from __future__ import annotations

import json
import random
import re
from typing import List, Optional

from .config import settings
from .tracing import Trace
from .types import RetrievedChunk

RUBRIC = (
    "You are a strict evaluator of a retrieval-augmented answer. Given the "
    "QUESTION, the retrieved CONTEXT, and the ANSWER, return a compact JSON "
    'object: {"faithfulness": <0..1>, "answer_relevance": <0..1>, '
    '"rationale": "<one sentence>"}. faithfulness = fraction of the answer that '
    "is supported by the context (1.0 = fully grounded, 0.0 = fabricated). "
    "answer_relevance = how well the answer addresses the question. "
    "Return ONLY the JSON."
)


def _parse_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class Judge:
    def __init__(self) -> None:
        self.enabled = bool(settings.gemini_api_key)
        self._client = None
        if self.enabled:
            try:
                from google import genai

                self._client = genai.Client(api_key=settings.gemini_api_key)
            except Exception:  # noqa: BLE001
                self.enabled = False

    def maybe_judge(
        self,
        question: str,
        answer: str,
        chunks: List[RetrievedChunk],
        trace: Trace,
        rng: Optional[random.Random] = None,
    ) -> Optional[dict]:
        rng = rng or random
        if not self.enabled or rng.random() > settings.judge_sample_rate:
            return None
        context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
        prompt = f"{RUBRIC}\n\nQUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER: {answer}"
        with trace.span("eval.judge") as sp:
            try:
                from google.genai import types

                resp = self._client.models.generate_content(
                    model=settings.gemini_judge_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                parsed = _parse_json(resp.text or "") or {}
            except Exception as exc:  # noqa: BLE001
                sp.set("error", f"{type(exc).__name__}: {exc}")
                return None
            faith = _clip(parsed.get("faithfulness"))
            rel = _clip(parsed.get("answer_relevance"))
            sp.set("model", settings.gemini_judge_model)
            sp.set("faithfulness", faith)
            sp.set("answer_relevance", rel)
            sp.set("rationale", parsed.get("rationale", ""))
        trace.set("faithfulness", faith)
        trace.set("answer_relevance", rel)
        return {"faithfulness": faith, "answer_relevance": rel,
                "rationale": parsed.get("rationale", "")}


def _clip(v) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None
