"""Pluggable tokenizer abstraction.

The default implementation uses ``tiktoken`` with the ``cl100k_base`` encoding
(works well for general Chinese+English text and is the de-facto fallback for
non-OpenAI models). If ``tiktoken`` cannot be imported (e.g. in a slim test
environment), we degrade gracefully to a character-count heuristic so the
context manager still produces a deterministic, conservative budget.
"""
from __future__ import annotations

from typing import Protocol

from app.core.logging import get_logger

_logger = get_logger(__name__)


class Tokenizer(Protocol):
    """Anything that can count tokens for a text."""

    def count(self, text: str) -> int: ...


class TiktokenTokenizer:
    """Counts tokens via ``tiktoken``. Lazily loaded; safe to instantiate even
    when tiktoken is missing — the constructor will record a warning and the
    object then behaves like a CharFallbackTokenizer.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._enc = None
        try:
            import tiktoken  # type: ignore

            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception as e:  # noqa: BLE001
            _logger.warning(
                "tokenizer.tiktoken_unavailable",
                error=str(e),
                fallback="char_heuristic",
            )

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is None:
            return _char_heuristic(text)
        return len(self._enc.encode(text))


class CharFallbackTokenizer:
    """Cheap heuristic: 0.5 tokens per character.

    Conservative for mixed CJK+ASCII corpora; intentionally over-estimates
    Chinese (1 char ≈ 1 token) but under-counts ASCII so the resulting cap is
    a soft upper bound rather than a precise count.
    """

    def count(self, text: str) -> int:
        return _char_heuristic(text)


def _char_heuristic(text: str) -> int:
    if not text:
        return 0
    # Treat each CJK character as ~1 token, each ASCII as ~0.25 tokens.
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    ascii_like = len(text) - cjk
    return int(cjk + ascii_like * 0.25) + 1


def get_default_tokenizer() -> Tokenizer:
    return TiktokenTokenizer()
