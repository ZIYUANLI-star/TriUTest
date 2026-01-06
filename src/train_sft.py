import argparse, yaml, os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from src.data.json_dataset import load_supervised_dataset
from src.data.collator import SFTDataCollator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    base_model = cfg["base_model"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=cfg["training"].get("use_4bit", True),
        torch_dtype="auto",
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"]
    )
    model = get_peft_model(model, lora_cfg)

    expansion_files = []
    if cfg.get("add_domain_expansion", False):
        # 可把 data/domain_expansion 下的文件加入
        import glob
        expansion_files = glob.glob("data/domain_expansion/*.jsonl")

    ds = load_supervised_dataset(cfg["train_file"], cfg.get("add_domain_expansion", False), expansion_files)

    collator = SFTDataCollator(tokenizer, max_len=cfg["training"]["max_len"])

    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg["training"]["micro_batch_size"],
        gradient_accumulation_steps=cfg["training"]["grad_acc_steps"],
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=float(cfg["training"]["lr"]),
        warmup_ratio=cfg["training"]["warmup_ratio"],
        logging_steps=10,
        save_steps=200,
        bf16=cfg["training"]["bf16"],
        optim="paged_adamw_32bit",
        report_to="none",
        remove_unused_columns = False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
