from __future__ import annotations

import json
import random
from typing import List, Literal

TextDomain = Literal["english", "chinese", "mixed", "code", "markdown", "json", "emoji"]


def _lorem_words() -> List[str]:
    return (
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore "
        "magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
        "consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
        "pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est "
        "laborum"
    ).split()


def _zh_phrases() -> List[str]:
    return [
        "大模型",
        "分词器",
        "上下文",
        "前缀缓存",
        "实时语音",
        "端侧推理",
        "延迟",
        "吞吐",
        "鲁棒性",
        "一致性检验",
        "代码助手",
        "系统提示词",
        "工具调用",
        "长对话",
        "越聊越慢",
        "功耗发热",
        "性能瓶颈",
    ]


def _emoji() -> List[str]:
    return ["😀", "🚀", "✨", "🔥", "📉", "📈", "🧪", "🧩", "🧠", "⚡", "⏱️", "🔁", "✅", "❌"]


def _code_lines() -> List[str]:
    return [
        "def f(x):\n    return x * 2\n",
        "for i in range(10):\n    print(i)\n",
        "const sum = (a, b) => a + b;\n",
        "if (err) { throw err; }\n",
        "SELECT id, name FROM users WHERE id = 42;\n",
        "curl -X POST https://api.example.com/v1/chat -d '{\"q\":\"hi\"}'\n",
        "class Node:\n    def __init__(self, v):\n        self.v = v\n",
    ]


def _md_blocks() -> List[str]:
    return [
        "# Title\n\n",
        "## Section\n\n",
        "- item 1\n- item 2\n- item 3\n\n",
        "> blockquote\n\n",
        "```python\nprint('hello')\n```\n\n",
        "`inline code` and **bold**.\n\n",
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n",
    ]


def make_text(char_target: int, *, seed: int, domain: TextDomain, header: str = "") -> str:
    rng = random.Random(seed)
    parts: List[str] = []
    if header:
        parts.append(header.rstrip() + "\n")

    while sum(len(p) for p in parts) < char_target:
        if domain == "english":
            words = _lorem_words()
            line = " ".join(rng.choice(words) for _ in range(rng.randint(8, 20))) + "\n"
        elif domain == "chinese":
            ph = _zh_phrases()
            line = "".join(rng.choice(ph) for _ in range(rng.randint(6, 14))) + "。\n"
        elif domain == "emoji":
            em = _emoji()
            words = _lorem_words()
            line = (
                "".join(rng.choice(em) for _ in range(rng.randint(1, 4)))
                + " "
                + " ".join(rng.choice(words) for _ in range(rng.randint(5, 12)))
                + "\n"
            )
        elif domain == "code":
            line = rng.choice(_code_lines())
        elif domain == "markdown":
            line = rng.choice(_md_blocks())
        elif domain == "json":
            obj = {
                "id": rng.randint(1, 10_000),
                "query": " ".join(rng.choice(_lorem_words()) for _ in range(rng.randint(6, 14))),
                "tags": rng.sample(_zh_phrases(), k=rng.randint(2, 6)),
                "ok": True,
            }
            line = json.dumps(obj, ensure_ascii=False) + "\n"
        elif domain == "mixed":
            # Mix english/chinese/emoji/code-ish tokens.
            pick = rng.choice(["english", "chinese", "emoji", "code"])
            line = make_text(200, seed=rng.randint(0, 2**31 - 1), domain=pick, header="")
        else:
            raise ValueError(f"unknown domain: {domain}")

        parts.append(line)

    text = "".join(parts)
    return text[:char_target]


def make_suffixes(n: int, char_target: int, *, seed: int, domain: TextDomain) -> List[str]:
    rng = random.Random(seed)
    out: List[str] = []
    for i in range(n):
        out.append(make_text(char_target, seed=rng.randint(0, 2**31 - 1), domain=domain, header=f"User#{i}: "))
    return out


def make_chat_deltas(turns: int, chars_per_turn: int, *, seed: int, domain: TextDomain) -> List[str]:
    rng = random.Random(seed)
    out: List[str] = []
    for i in range(turns):
        user = make_text(
            chars_per_turn,
            seed=rng.randint(0, 2**31 - 1),
            domain=domain,
            header=f"User({i}): ",
        )
        assistant = make_text(
            chars_per_turn,
            seed=rng.randint(0, 2**31 - 1),
            domain=domain,
            header=f"Assistant({i}): ",
        )
        out.append("\n" + user + assistant)
    return out

