#这段函数是把你的训练数据读进来并标准化，最后返回一个 HuggingFace datasets.Dataset 对象，供 SFT/GKD 训练用。

from typing import Dict, Any, List
from datasets import Dataset
import random, json

def load_supervised_dataset(path: str, add_domain_expansion: bool=False, expansion_files: List[str]=None) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)

    rows = []
    for ex in arr:
        # 支持你给的结构： prompt / chosen / response
        prompt = ex["prompt"]
        target = ex.get("chosen") or ex.get("response") or ""
        rows.append({"prompt": prompt, "target": target})

    if add_domain_expansion and expansion_files:
        for ef in expansion_files:
            with open(ef, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        j = json.loads(line)
                        rows.append(j)

    return Dataset.from_list(rows)
