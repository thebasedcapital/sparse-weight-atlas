"""Build ``runs/<model>/vocab.json``: decoded token strings plus the contract classes.

Each id 0..2047 is decoded on its own with the tokenizer's ByteLevel decoder
(``Tokenizer.decode([i], skip_special_tokens=False)``), keeping the exact string.  Class
membership is then a pure function of that string (CONTRACT.md "Vocab classes").
"""

from __future__ import annotations

import argparse
import json
import keyword
import string
from pathlib import Path

from tokenizers import Tokenizer

from swa.model import ROOT, TOKENIZER_PATH

VOCAB_SIZE = 2048
EOT_ID = 2047

_ASCII_DIGITS = set(string.digits)
_PUNCT = set(string.punctuation)
_OPERATOR_CHARS = set("+-*/%=<>!&|^~")
_OPEN = set("([{")
_CLOSE = set(")]}")


def decoded_tokens(tokenizer_path: Path = TOKENIZER_PATH) -> list[str]:
    tok = Tokenizer.from_file(str(tokenizer_path))
    return [tok.decode([i], skip_special_tokens=False) for i in range(VOCAB_SIZE)]


def classes_for(tokens: list[str]) -> dict[str, list[int]]:
    def ids(pred) -> list[int]:
        return [i for i, s in enumerate(tokens) if pred(s)]

    return {
        "digit_only": ids(lambda s: len(s) > 0 and all(c in _ASCII_DIGITS for c in s)),
        "has_digit": ids(lambda s: any(c in _ASCII_DIGITS for c in s)),
        "alpha_only": ids(lambda s: len(s) > 0 and all(c.isalpha() for c in s)),
        "upper_start": ids(lambda s: len(s) > 0 and s[0].isupper()),
        "lower_only": ids(lambda s: len(s) > 0 and all(c.islower() for c in s)),
        "space_lead": ids(lambda s: s.startswith(" ")),
        "newline": ids(lambda s: "\n" in s),
        "indent": ids(lambda s: len(s) > 0 and all(c.isspace() for c in s)),
        "punct_only": ids(lambda s: len(s) > 0 and all(c in _PUNCT for c in s)),
        "quote": ids(lambda s: "'" in s or '"' in s),
        "bracket_open": ids(lambda s: any(c in _OPEN for c in s)),
        "bracket_close": ids(lambda s: any(c in _CLOSE for c in s)),
        "operator": ids(lambda s: len(s) > 0 and all(c in _OPERATOR_CHARS for c in s)),
        "dot": ids(lambda s: "." in s),
        "comma": ids(lambda s: "," in s),
        "colon": ids(lambda s: ":" in s),
        "underscore": ids(lambda s: "_" in s),
        "keyword": ids(lambda s: len(s.strip()) > 0 and keyword.iskeyword(s.strip())),
        "eot": [EOT_ID],
    }


def build(model_name: str) -> Path:
    tokens = decoded_tokens()
    out = ROOT / "runs" / model_name / "vocab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tokens": tokens, "classes": classes_for(tokens)}
    out.write_text(json.dumps(payload, ensure_ascii=False))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="csp_yolo1")
    args = ap.parse_args(argv)
    path = build(args.model)
    data = json.loads(path.read_text())
    print(f"wrote {path}")
    for name, ids in data["classes"].items():
        print(f"  {name:14s} {len(ids):5d}")
    print(f"  tokens {len(data['tokens'])}")


if __name__ == "__main__":
    main()
