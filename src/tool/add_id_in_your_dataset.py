# tools/add_ids.py
import json, argparse, os, tempfile, shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="输入 JSON 文件（数组）")
    ap.add_argument("--out", dest="out", default=None, help="输出 JSON 文件（默认在原文件名后加 _with_id）")
    ap.add_argument("--prefix", default="ex_", help="id 前缀，默认 ex_")
    ap.add_argument("--start", type=int, default=1, help="起始编号（含），默认 1")
    ap.add_argument("--width", type=int, default=5, help="编号宽度，默认 5（ex_00001）")
    ap.add_argument("--force", action="store_true", help="若已存在 id 字段也强制覆盖")
    ap.add_argument("--inplace", action="store_true", help="原地修改（会先写临时文件再替换，创建 .bak 备份）")
    args = ap.parse_args()

    in_path = args.inp
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    # 读取（用 utf-8 避免 Windows 下 gbk 问题；带 BOM 也能正常解析）
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "输入 JSON 须为数组(list)"

    # 先收集已有 id，避免冲突
    seen = set()
    for ex in data:
        if isinstance(ex, dict) and "id" in ex and not args.force:
            seen.add(str(ex["id"]))

    # 生成 id，必要时跳过已被占用的
    cur = args.start
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            raise ValueError(f"第 {i} 条不是对象：{type(ex)}")
        if "id" in ex and not args.force:
            # 保留原 id
            continue
        # 生成一个未被占用的 id
        while True:
            cand = f"{args.prefix}{str(cur).zfill(args.width)}"
            cur += 1
            if cand not in seen:
                seen.add(cand)
                ex["id"] = cand
                break

    # 决定输出位置
    if args.inplace:
        out_path = in_path
    else:
        if args.out:
            out_path = args.out
        else:
            root, ext = os.path.splitext(in_path)
            out_path = root + "_with_id" + (ext or ".json")

    # 写文件（原子替换；保留 .bak 备份以防万一）
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="add_ids_", suffix=".json", dir=os.path.dirname(out_path) or ".")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if args.inplace:
            bak = out_path + ".bak"
            shutil.copy2(out_path, bak) if os.path.exists(out_path) else None
        shutil.move(tmp_path, out_path)
        print(f"✅ 写入完成：{out_path}")
        if args.inplace and os.path.exists(out_path + ".bak"):
            print(f"（已备份原文件到 {out_path+'.bak'}）")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    main()
