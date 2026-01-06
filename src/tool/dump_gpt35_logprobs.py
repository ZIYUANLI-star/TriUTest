# tools/dump_gpt35_logprobs.py
import os, json, time, argparse
from openai import OpenAI
import tiktoken

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="your_dataset_with_id.json（每条要有唯一id, 包含 prompt/target 或 prompt+chosen）")
    ap.add_argument("--out_jsonl", required=True, help="输出 teacher_logprobs.jsonl")
    ap.add_argument("--model", default="gpt-3.5-turbo")  # 你账号可用的 chat 模型
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--delay", type=float, default=0.3, help="每次请求之间的停顿，防止限流")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ.get("your_api"))
    enc = tiktoken.encoding_for_model(args.model)

    data = load_json(args.in_json)
    out_rows = []

    for ex in data:
        ex_id = ex.get("id") or ex.get("uid") or ex.get("sample_id")
        assert ex_id is not None, "请给每条样本一个稳定 id"
        prompt = ex["prompt"]
        target = ex.get("chosen") or ex.get("response") or ex.get("target") or ""
        if not target:
            out_rows.append({"id": ex_id, "t_logprobs": []})
            continue

        # 用 tokenizer 得到目标 token 序列
        target_tokens = enc.encode(target)

        # 我们把对话历史构造成：
        #   system: 角色提示
        #   user:   原始 prompt（你训练时给学生的 prompt）
        #   assistant: 已经“生成”的目标前缀（逐步扩展）
        #
        # 每一步只生成 1 个 token，并用 logit_bias 强制指定的 token。
        assistant_prefix_tokens = []
        t_logprobs = []

        for i, tok in enumerate(target_tokens):
            # 已有的 assistant 前缀（把前面的 token 还原成字符串，放到历史里）
            assistant_prefix_text = enc.decode(assistant_prefix_tokens) if assistant_prefix_tokens else ""
            messages = [
                {"role": "system", "content": args.system},
                {"role": "user", "content": prompt},
            ]
            if assistant_prefix_text:
                messages.append({"role": "assistant", "content": assistant_prefix_text})

            # 强制当前 token
            logit_bias = { str(tok): 100 }  # 100 足够大，一般可强制为该 token
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=0,   # 不需要候选，只要 chosen 的 logprob
                logit_bias=logit_bias,
            )

            # 取出生成的这个 token 的 logprob
            item = resp.choices[0].logprobs.content[0]  # 单个 token
            # 有的模型可能会把 token 拆成不同字节序列，按返回字段兼容
            logprob = float(item.logprob)
            t_logprobs.append(logprob)

            # 把这次生成的 token 拼到前缀，进入下一步
            assistant_prefix_tokens.append(tok)

            time.sleep(args.delay)

        out_rows.append({"id": ex_id, "t_logprobs": t_logprobs})

    save_jsonl(args.out_jsonl, out_rows)

if __name__ == "__main__":
    main()
