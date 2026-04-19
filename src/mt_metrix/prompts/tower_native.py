"""Tower-native QE prompt (Alves et al. 2024, "Tower: An Open Multilingual Large Language Model for Translation-Related Tasks").

The Tower paper lists an instruction-tuned prompt template for
quality estimation tasks. This template closely mirrors GEMBA-DA but uses
the Tower chat framing (system / user role). vLLM automatically applies the
model's chat template when you pass ``messages=`` or a list of dicts.

Use this prompt by setting ``params.prompt_mode: tower-native`` in the run
config. We parse the response the same way as GEMBA-DA (look for an integer
in [0, 100]).
"""
from __future__ import annotations

from mt_metrix.prompts.gemba_da import _language_name, parse_gemba_da_score

__all__ = ["build_tower_native_messages", "parse_tower_native_score"]


def build_tower_native_messages(source: str, target: str, lang_pair: str) -> list[dict[str, str]]:
    """Chat-format messages for Tower QE.

    Returns a two-turn exchange: a system instruction and a user turn
    containing the source and target. The Tower chat template is applied by
    the tokeniser (vLLM supports this via ``apply_chat_template``).
    """
    if "-" in lang_pair:
        src_code, tgt_code = lang_pair.split("-", 1)
        src_name, tgt_name = _language_name(src_code), _language_name(tgt_code)
    else:
        src_name, tgt_name = "source", "target"

    system = (
        "You are an expert evaluator of machine translation quality. "
        "You rate translations on a scale from 0 (worst) to 100 (best) based on "
        "semantic accuracy, fluency, and faithfulness to the source."
    )
    user = (
        f"{src_name} source: \"{source}\"\n"
        f"{tgt_name} translation: \"{target}\"\n\n"
        "Respond with only the numeric score (0-100)."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# parsing is identical to GEMBA-DA
parse_tower_native_score = parse_gemba_da_score
