from __future__ import annotations

import base64
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from .config import Settings


@dataclass(frozen=True)
class AnalysisResult:
    keep: bool
    title: str
    summary: str
    tags: list[str]
    confidence: float
    start_offset_seconds: int
    end_offset_seconds: int
    raw: dict[str, Any]

    def should_save(self, threshold: float) -> bool:
        # The model returns both keep (boolean intent) and confidence
        # (0..1). Both are required: a low-confidence "keep=true" from a
        # 2B vision model on a low-res contact sheet is unreliable - it
        # tends to see toys/furniture and assume a child is present. An
        # AND gate makes the threshold filter out those weak guesses.
        return self.keep and self.confidence >= threshold


def _image_content(path: Path) -> dict[str, Any]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{data}",
        },
    }


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("model response did not contain a JSON object")
    return json.loads(match.group(0))


def _coerce_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "save", "keep"}
    return bool(value)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _post_json(endpoint: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama.cpp request failed: HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"llama.cpp request failed: {exc.reason}") from exc


class LlamaAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def analyze(self, *, video_path: Path, image_paths: list[Path], duration_seconds: int) -> AnalysisResult:
        if not image_paths:
            raise ValueError("no sampled images available for analysis")

        endpoint = self.settings.llama_base_url.rstrip("/") + "/chat/completions"
        image_note = (
            "The image is a contact sheet. Read cells chronologically from left to right, "
            "then top to bottom."
            if self.settings.analysis_image_mode == "contact_sheet"
            else "The images are sampled video frames in chronological order."
        )
        instructions = (
            f"{self.settings.analysis_prompt}\n\n"
            f"{image_note}\n\n"
            "Return exactly one JSON object with these fields:\n"
            "- keep: boolean\n"
            "- title: short human title\n"
            "- summary: one or two sentences\n"
            "- tags: array of short strings\n"
            "- confidence: number from 0 to 1\n"
            "- start_offset_seconds: integer offset inside the segment\n"
            "- end_offset_seconds: integer offset inside the segment\n\n"
            "Tighten start_offset_seconds and end_offset_seconds around the actual highlight; "
            "do not span the whole segment unless every frame is a genuine moment.\n\n"
            "Be strict about keep=false: an empty or quiet room with no child visible is "
            "keep=false regardless of furniture, toys, lighting, or audio. If you are unsure "
            "whether my daughter is visible and active, set keep=false and confidence below 0.5. "
            "Only set keep=true when you can clearly identify my daughter in motion or in a "
            "genuine interaction.\n\n"
            f"Segment file: {video_path.name}\n"
            f"Segment duration seconds: {duration_seconds}\n"
            f"Number of provided images: {len(image_paths)}"
        )

        user_content: list[dict[str, Any]] = [{"type": "text", "text": instructions}]
        user_content.extend(_image_content(path) for path in image_paths)

        headers = {"Content-Type": "application/json"}
        if self.settings.llama_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llama_api_key}"

        payload = {
            "model": self.settings.llama_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful family video curator. Return JSON only.",
                },
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        body = await asyncio.to_thread(
            _post_json,
            endpoint,
            headers,
            payload,
            self.settings.llama_timeout_seconds,
        )

        content = body["choices"][0]["message"]["content"]
        data = _extract_json(_extract_message_text(content))

        confidence = max(0.0, min(_coerce_float(data.get("confidence"), 0.0), 1.0))
        start_offset = max(0, _coerce_int(data.get("start_offset_seconds"), 0))
        end_offset = _coerce_int(data.get("end_offset_seconds"), duration_seconds)
        end_offset = max(start_offset + 1, min(end_offset, duration_seconds))

        title = str(data.get("title") or "Family moment").strip()[:120]
        summary = str(data.get("summary") or "Saved by the family moment analyzer.").strip()

        return AnalysisResult(
            keep=_coerce_bool(data.get("keep")),
            title=title,
            summary=summary,
            tags=_coerce_tags(data.get("tags")),
            confidence=confidence,
            start_offset_seconds=start_offset,
            end_offset_seconds=end_offset,
            raw=data,
        )
