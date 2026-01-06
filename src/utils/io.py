# 读写json建目录

import json, os, sys, math, random, time, pathlib
from typing import List, Dict, Any, Iterable

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def ensure_dir(d: str):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
