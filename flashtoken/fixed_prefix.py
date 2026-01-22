from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class StableSplit:
    stable_tokens: List[int]
    stable_text: str
    unstable_text: str


def stable_split_for_text(encoding, text: str) -> StableSplit:
    """
    Split `text` into:
      - stable_tokens: a token prefix guaranteed not to change if more text is appended
      - unstable_text: the remaining suffix (may need re-tokenization when text grows)

    This uses `tiktoken.Encoding.encode_with_unstable`, which is intended for
    incremental / streaming tokenization.
    """
    stable_tokens, _completions = encoding.encode_with_unstable(text)
    stable_text = encoding.decode(stable_tokens)
    if not text.startswith(stable_text):
        raise ValueError("Stable token prefix does not decode to a string prefix of the input text.")
    return StableSplit(
        stable_tokens=list(stable_tokens),
        stable_text=stable_text,
        unstable_text=text[len(stable_text) :],
    )


class FixedPrefixTokenCache:
    """
    Exact tokenization for many strings of the form: `prefix + suffix`.

    It caches the *stable* token prefix of `prefix`, then per request only encodes
    the small `(unstable_prefix_tail + suffix)` tail.

    Intended for: long fixed system prompt / template reuse.
    """

    def __init__(self, encoding, prefix: str) -> None:
        self._encoding = encoding
        self._prefix = prefix
        self._split = stable_split_for_text(encoding, prefix)

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def stable_prefix_token_count(self) -> int:
        return len(self._split.stable_tokens)

    @property
    def unstable_prefix_char_count(self) -> int:
        return len(self._split.unstable_text)

    def encode_ordinary(self, suffix: str) -> List[int]:
        return self._split.stable_tokens + self.encode_ordinary_tail(suffix)

    def encode_ordinary_tail(self, suffix: str) -> List[int]:
        tail = self._split.unstable_text + suffix
        return self._encoding.encode_ordinary(tail)

