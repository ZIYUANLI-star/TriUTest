"""Collator for DQ-GKD: gold CE batch + on-policy candidates with
test-sensitive span masks (paper Sec. 3.2 (1)) and quality/diversity weights.
"""
import ast, re
from dataclasses import dataclass
from typing import Dict, List

import torch


def sensitive_char_ranges(code: str):
    """Char ranges of test-sensitive segments: input construction, UUT invocation,
    assertion expressions, expected exceptions. Falls back to regex on parse error."""
    ranges = []
    try:
        tree = ast.parse(code)
        lines = code.splitlines(keepends=True)
        line_off = [0]
        for ln in lines:
            line_off.append(line_off[-1] + len(ln))

        def node_range(node):
            try:
                s = line_off[node.lineno - 1] + node.col_offset
                e = line_off[node.end_lineno - 1] + node.end_col_offset
                return (s, e)
            except Exception:
                return None

        for node in ast.walk(tree):
            hit = False
            if isinstance(node, ast.Assert):
                hit = True
            elif isinstance(node, ast.With):
                src = ast.dump(node.items[0].context_expr) if node.items else ""
                if "raises" in src:
                    hit = True
            elif isinstance(node, (ast.Assign, ast.Expr)):
                v = node.value if hasattr(node, "value") else None
                if isinstance(v, ast.Call):
                    hit = True   # invocation / input construction
            if hit:
                r = node_range(node)
                if r:
                    ranges.append(r)
    except SyntaxError:
        for m in re.finditer(r"^.*(assert|raises|\w+\().*$", code, re.M):
            ranges.append((m.start(), m.end()))
    return ranges


@dataclass
class DQGKDCollator:
    tokenizer: any
    max_len: int = 2048
    cand_max_len: int = 1536

    def _encode_gold(self, prompt: str, target: str):
        tok_p = self.tokenizer(prompt, add_special_tokens=False)
        tgt = "\n" + target if target and not target.startswith("\n") else target
        tok_t = self.tokenizer(tgt, add_special_tokens=False)
        ids_p, ids_t = tok_p["input_ids"], tok_t["input_ids"]
        # 目标末尾显式追加 EOS（与 SFT collator 一致）
        if self.tokenizer.eos_token_id is not None and \
                (not ids_t or ids_t[-1] != self.tokenizer.eos_token_id):
            ids_t = ids_t + [self.tokenizer.eos_token_id]
        space = self.max_len
        keep_t = min(len(ids_t), space)
        keep_p = max(0, min(len(ids_p), space - keep_t))
        ids = ids_p[:keep_p] + ids_t[:keep_t]
        labels = [-100] * keep_p + ids_t[:keep_t]
        return ids, labels

    def _encode_candidate(self, prompt: str, cand: str):
        """Returns ids, span_mask over the *candidate* segment (prompt masked out)."""
        tok_p = self.tokenizer(prompt, add_special_tokens=False)
        c = "\n" + cand if cand and not cand.startswith("\n") else cand
        enc = self.tokenizer(c, add_special_tokens=False, return_offsets_mapping=True)
        ids_c, offs = enc["input_ids"], enc["offset_mapping"]
        ranges = sensitive_char_ranges(cand)
        # shift because of the prepended "\n"
        shift = 1 if c is not cand else 0
        mask_c = []
        for (s, e) in offs:
            s -= shift
            e -= shift
            m = 0
            for (rs, re_) in ranges:
                if s < re_ and e > rs:
                    m = 1
                    break
            mask_c.append(m)
        if not any(mask_c):
            mask_c = [1] * len(ids_c)   # degenerate candidate: distill uniformly
        space = self.cand_max_len
        keep_c = min(len(ids_c), space)
        keep_p = max(0, min(len(tok_p["input_ids"]), space - keep_c))
        ids = tok_p["input_ids"][:keep_p] + ids_c[:keep_c]
        mask = [0] * keep_p + mask_c[:keep_c]
        return ids, mask

    def __call__(self, batch: List[Dict]):
        pad_id = self.tokenizer.pad_token_id
        gold_ids, gold_labels = [], []
        cand_ids, cand_masks, cand_w = [], [], []
        for ex in batch:
            ids, labels = self._encode_gold(ex["prompt"], ex["target"])
            gold_ids.append(torch.tensor(ids, dtype=torch.long))
            gold_labels.append(torch.tensor(labels, dtype=torch.long))
            for cand, w in zip(ex.get("cands", []), ex.get("cand_weights", [])):
                cids, cmask = self._encode_candidate(ex["prompt"], cand)
                cand_ids.append(torch.tensor(cids, dtype=torch.long))
                cand_masks.append(torch.tensor(cmask, dtype=torch.float))
                cand_w.append(float(w))

        def pad(seqs, val):
            return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=val)

        out = {
            "input_ids": pad(gold_ids, pad_id),
            "attention_mask": pad([torch.ones_like(x) for x in gold_ids], 0),
            "labels": pad(gold_labels, -100),
        }
        if cand_ids:
            out["cand_input_ids"] = pad(cand_ids, pad_id)
            out["cand_attention_mask"] = pad([torch.ones_like(x) for x in cand_ids], 0)
            out["cand_span_mask"] = pad(cand_masks, 0.0)
            out["cand_weight"] = torch.tensor(cand_w, dtype=torch.float)
        return out
