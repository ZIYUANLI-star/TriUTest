from dataclasses import dataclass
from typing import Dict, List
import torch

@dataclass
class SFTDataCollator:
    tokenizer: any
    max_len: int = 2048

    def __call__(self, batch: List[Dict]):
        input_ids_list, attn_list, labels_list = [], [], []
        for ex in batch:
            prompt = ex["prompt"]
            target = ex["target"]

            # 分别分词（不加特殊符号，避免重复 bos/eos）
            tok_prompt = self.tokenizer(prompt, add_special_tokens=False)
            tok_target = self.tokenizer("\n" + target if target and not target.startswith("\n") else target,
                                        add_special_tokens=False)

            ids_p = tok_prompt["input_ids"]
            ids_t = tok_target["input_ids"]

            # 预算：保留 target，尽量截断 prompt
            space = self.max_len
            keep_t = min(len(ids_t), space)
            space -= keep_t
            keep_p = max(0, min(len(ids_p), space))

            ids = ids_p[:keep_p] + ids_t[:keep_t]
            # attention mask
            attn = [1]*len(ids)

            # labels：prompt 部分忽略，target 部分监督，padding 忽略
            labels = [-100]*keep_p + ids_t[:keep_t]

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            attn_list.append(torch.tensor(attn, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        # pad 到批内最大长度
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attn_list, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
