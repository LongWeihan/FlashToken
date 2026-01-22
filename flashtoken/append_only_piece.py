from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import regex


@dataclass(frozen=True)
class TokenDelta:
    rollback_tokens: int
    tokens_to_append: List[int]


class AppendOnlyPieceTokenCache:
    """
    Incremental tokenizer cache for *append-only* prompts, based on pre-tokenization pieces.

    How tiktoken-style encodings work:
      1) split text by a regex into "pieces"
      2) BPE-encode each piece independently

    Key observation:
      When you append text, almost all old pieces stay unchanged; typically only the last
      1~few pieces can be re-segmented by the regex.

    This cache reprocesses only the last `backtrack_pieces` pieces on each append,
    returning a delta: (rollback_tokens, tokens_to_append). This shape is convenient
    for KV-cached inference: rollback last N KV entries then append new ones.

    Notes:
    - This uses tiktoken internals (`encoding._pat_str`, `encoding._encode_single_piece`)
      for performance and exactness.
    - This matches `encode_ordinary` semantics (no special tokens).
    """

    def __init__(self, encoding, initial_text: str = "", *, backtrack_pieces: int = 2) -> None:
        if backtrack_pieces < 1:
            raise ValueError("backtrack_pieces must be >= 1")
        self._encoding = encoding
        self._pattern = regex.compile(encoding._pat_str)
        self._backtrack_pieces = backtrack_pieces

        self._text = ""
        self._pieces: List[Tuple[int, int]] = []
        self._piece_tokens: List[List[int]] = []
        self._tokens: List[int] = []

        self.reset(initial_text)

    @property
    def text(self) -> str:
        return self._text

    @property
    def tokens(self) -> List[int]:
        return self._tokens

    @property
    def backtrack_pieces(self) -> int:
        return self._backtrack_pieces

    def reset(self, text: str = "") -> None:
        self._text = text
        self._pieces = []
        self._piece_tokens = []
        self._tokens = []

        for m in self._pattern.finditer(text):
            piece_text = m.group(0)
            piece_tokens = list(self._encoding._encode_single_piece(piece_text))
            self._pieces.append((m.start(), m.end()))
            self._piece_tokens.append(piece_tokens)
            self._tokens.extend(piece_tokens)

    def append_ordinary(self, delta: str) -> TokenDelta:
        if not delta:
            return TokenDelta(rollback_tokens=0, tokens_to_append=[])

        prev_piece_count = len(self._pieces)
        prev_token_count = len(self._tokens)

        self._text += delta

        if prev_piece_count == 0:
            self.reset(self._text)
            return TokenDelta(rollback_tokens=prev_token_count, tokens_to_append=list(self._tokens))

        b = min(self._backtrack_pieces, prev_piece_count)
        start_piece_index = prev_piece_count - b
        reprocess_start = self._pieces[start_piece_index][0]

        old_tail_token_count = 0
        for toks in self._piece_tokens[start_piece_index:]:
            old_tail_token_count += len(toks)

        rollback = old_tail_token_count
        if rollback:
            del self._tokens[-rollback:]

        tail_text = self._text[reprocess_start:]
        new_pieces: List[Tuple[int, int]] = []
        new_piece_tokens: List[List[int]] = []
        tokens_to_append: List[int] = []

        for m in self._pattern.finditer(tail_text):
            piece_text = m.group(0)
            piece_tokens = list(self._encoding._encode_single_piece(piece_text))
            new_pieces.append((reprocess_start + m.start(), reprocess_start + m.end()))
            new_piece_tokens.append(piece_tokens)
            tokens_to_append.extend(piece_tokens)

        self._pieces = self._pieces[:start_piece_index] + new_pieces
        self._piece_tokens = self._piece_tokens[:start_piece_index] + new_piece_tokens
        self._tokens.extend(tokens_to_append)

        return TokenDelta(rollback_tokens=rollback, tokens_to_append=tokens_to_append)

