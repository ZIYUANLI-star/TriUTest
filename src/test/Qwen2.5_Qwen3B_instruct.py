import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === 1) 改成你的本地快照路径（注意用 r"..." 原始字符串防止转义）===
MODEL_DIR = r"C:\Users\28772\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\Qwen2.5-3B"

# 一个示例提示词
USER_PROMPT1 = "Based on the function description and the code snippet below, please generate a comprehensive set of detailed test cases that cover typical usage, edge cases, and potential error conditions.\n\nFunction Description:\nSumming values in a dictionary.\n\n    Parameters\n    ----------\n    dictTotal : TYPE\n        DESCRIPTION.\n\n    Returns\n    -------\n    TYPE\n        DESCRIPTION.\n\nCode Under Test:\ndef SumTotals(dictTotal):\n    \n    \n    totalVal = 0.0\n    for keyVal in dictTotal.keys():\n        for keyVal2 in dictTotal[keyVal].keys():\n            totalVal += dictTotal[keyVal][keyVal2]\n    \n    return round(totalVal, 2)\n\n"
USER_PROMPT ="""I will provide you with the code under test and an existing test. Please modify the test code according to the error message so that the tests can be collected and executed successfully.
Code under test:
def SumTotals(dictTotal):
    totalVal = 0.0
    for keyVal in dictTotal.keys():
        for keyVal2 in dictTotal[keyVal].keys():
            totalVal += dictTotal[keyVal][keyVal2]
    return round(totalVal, 2)
    
test code:
def test():
    assert SumTotals({'a': {'b': 1.0}, 'c': {'d': 7.0}) == 3.0
    
Error output:
D:\ANACONDA\envs\RLHF\python.exe "D:/pycharm/PyCharm Community Edition 2024.1.4/plugins/python-ce/helpers/pycharm/_jb_pytest_runner.py" --target test_cut.py::test 
Testing started at 14:36 ...
Launching pytest with arguments test_cut.py::test --no-header --no-summary -q in D:\postgradute_Learning\paper\exam\GKD

============================= test session starts =============================
collecting ... 
test_cut.py:None (test_cut.py)
D:\ANACONDA\envs\RLHF\Lib\site-packages\_pytest\python.py:498: in importtestmodule
    mod = import_path(
D:\ANACONDA\envs\RLHF\Lib\site-packages\_pytest\pathlib.py:587: in import_path
    importlib.import_module(module_name)
D:\ANACONDA\envs\RLHF\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
D:\ANACONDA\envs\RLHF\Lib\site-packages\_pytest\\assertion\rewrite.py:177: in exec_module
    source_stat, co = _rewrite_test(fn, self.config)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
D:\ANACONDA\envs\RLHF\Lib\site-packages\_pytest\assertion\rewrite.py:357: in _rewrite_test
    tree = ast.parse(source, filename=strfn)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
D:\ANACONDA\envs\RLHF\Lib\ast.py:54: in parse
    return compile(source, filename, mode, flags,
E     File "D:\postgradute_Learning\paper\exam\GKD\test_cut.py", line 11
E       assert SumTotals({'a': {'b': 1.0}, 'c': {'d': 7.0}) == 3.0
E                                                         ^
E   SyntaxError: closing parenthesis ')' does not match opening parenthesis '{'
collected 0 items / 1 error

============================== 1 error in 0.07s ===============================
ERROR: found no collectors for D:\postgradute_Learning\paper\exam\GKD\test_cut.py::test


Process finished with exit code 4

                                                  ^
"""


# === 2) 基础设置（尽量快 & 稳）===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# GPU 上优先用 fp16；CPU 上用 fp32
if torch.cuda.is_available():
    torch_dtype = torch.float16
    device_map = "auto"
else:
    torch_dtype = torch.float32
    device_map = {"": "cpu"}  # 全部放 CPU

# === 3) 离线加载 tokenizer 和 model ===
# local_files_only=True 确保不走网络；trust_remote_code=True 兼容 Qwen 的自定义代码
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR, use_fast=True, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch_dtype, device_map=device_map,
    trust_remote_code=True, local_files_only=True
)
model.eval()

# pad_token 兜底（有些模型没有 pad_token）
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

# === 4) 组装输入（优先使用 chat 模板；否则退化为纯文本）===
def build_input(prompt: str) -> str:
    # 如果是 Instruct/Chat 模型且提供了 chat_template，使用官方对话模板更稳
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # 没有模板就用纯文本
        return prompt

text_input = build_input(USER_PROMPT)

# === 5) 推理 ===
inputs = tokenizer(text_input, return_tensors="pt")
# 把张量放到模型所在设备（device_map="auto" 时模型可能被切片在多 GPU；to(model.device) 也能工作）
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        do_sample=True,           # 采样生成（更有多样性），如需确定性可改为 False
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# 只取新增的内容（去掉提示词本身）
gen_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

print("\n[Prompt]\n", USER_PROMPT)
print("\n[Model Output]\n", gen_text)
