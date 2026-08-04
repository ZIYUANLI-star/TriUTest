# TriUTest

TriUTest is a three-stage post-training framework for Python unit test generation with small open-weight LLMs. It combines:

1. **SFT** — supervised fine-tuning on prompt–test pairs to learn the basic generation format.
2. **DQ-GKD** — diversity- and quality-aware guided knowledge distillation: candidates sampled from the student are scored by executability, assertion adequacy, and failure type, then distilled from a larger teacher with quality-weighted multi-candidate likelihood and span-masked test-sensitive supervision.
3. **SM-GRPO** — stateful marginal group relative policy optimization: a per-UUT mutant coverage state turns raw kill rewards into *marginal incremental kill* rewards, combined with stability and assertion-adequacy regularizers under a symmetric KL constraint.

This repository contains the method implementation, the training/evaluation configurations, and the raw per-seed evaluation results of the TriUTest configurations reported in the paper.

## Repository layout

```
src/                     Method implementation
  train_sft.py             Stage 1: SFT entry point
  train_dq_gkd.py          Stage 2: DQ-GKD trainer (quality-weighted multi-candidate distillation)
  train_sm_grpo.py         Stage 3: SM-GRPO entry point
  data/                    Dataset loading and collators (SFT + DQ-GKD)
  losses/gkd.py            Generalized JSD distillation loss
  rl/grpo_trainer.py       GRPO rollout/update loop (winner-only, group-baseline)
  reward/
    sandbox.py             Sandboxed pytest / mutation execution
    quality.py             Candidate quality scoring (executability, assertions, AST features)
    rewarders.py           Composite reward (mutant kill, stability, assertion adequacy)
    rewarders_new.py       Reward component helpers
    stateful.py            Stateful marginal mutant-kill reward (SM-GRPO core)
  augment/robust_code.py   Test-code cleaning / salvage utilities
configs/                 Training configurations used in the paper
  sft_{1p5b,3b,7b,14b}.yaml
  gkd_{3b_14b,3b_7b,1p5b_14b,1p5b_7b}.yaml
  grpo_triu3b_sa.yaml      Main TriUTest configuration (3B student, 14B teacher, spec-anchored)
  grpo_*.yaml              Ablation / alternative student–teacher configurations
data_construction/       DQ-GKD candidate generation and scoring
scripts/                 Shell entry points for the three training stages
evaluation/              Benchmark preparation, test generation, and metric computation
  prep_benchmarks.py       Build UUT prompts from HumanEval / QuixBugs / codetiming / apimd
  gen_tests_vllm.py        Batch test generation with vLLM
  run_eval.py              Statement / branch coverage evaluation
  run_mkr.py               Mutation Kill Rate (MutPy, Python 3.9 environment)
  ftr_v2.py                Fail Trigger Rate on QuixBugs defective programs
  aggregate_stats.py       Cross-seed aggregation and statistics
results/                 Raw per-seed TriUTest results (seeds 40-44)
  eval/                    Coverage JSONs per subject and seed
  mkr/                     Mutation Kill Rate JSONs
  ftr_v2/                  Fail Trigger Rate JSONs (QuixBugs)
```

Result directories follow the naming `triutest-<student>[-t7b][-sa]`, e.g. `triutest-3b-sa` is the main configuration reported as *TriUTest* in the paper; `triutest-1p5b` and `triutest-3b-t7b*` are the alternative student–teacher pairings.

## Environment

- Dual evaluation environment as in the paper: coverage/FTR measurement and original-program test execution run under **Python 3.13**; a separate **Python 3.9** environment is required for MutPy-based mutation testing (`run_mkr.py`). The training code itself is compatible with Python 3.9+.
- CUDA GPU with bf16 support (the paper's experiments used a single NVIDIA A800 80GB GPU). In the main 3B x 14B configuration the teacher is loaded **frozen in bf16** (`teacher_4bit: false` in `configs/gkd_3b_14b.yaml`); the student uses LoRA on a 4-bit quantized base with bf16 compute, as specified in Table 5 of the paper.
- Base models: Qwen2.5-Instruct family (1.5B/3B/7B/14B).

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
# Stage 1: SFT
bash scripts/run_sft.sh configs/sft_3b.yaml

# Stage 2: DQ-GKD (generate + score candidates from the SFT student, then distill)
python data_construction/gen_gkd_candidates.py --model <sft-checkpoint> --k 4
python data_construction/score_gkd_candidates.py
bash scripts/run_distill_gkd.sh configs/gkd_3b_14b.yaml

# Stage 3: SM-GRPO
bash scripts/run_grpo_mutant.sh configs/grpo_triu3b_sa.yaml
```

## Evaluation

```bash
python evaluation/prep_benchmarks.py                  # build benchmark prompts
python evaluation/gen_tests_vllm.py --model <ckpt> --method triutest-3b-sa \
    --subjects humaneval quixbugs codetiming apimd --seeds 40 41 42 43 44
python evaluation/run_eval.py ...                     # statement/branch coverage
python evaluation/run_mkr.py ...                      # MKR (run inside the Python 3.9 env)
python evaluation/ftr_v2.py ...                       # FTR on QuixBugs
python evaluation/aggregate_stats.py                  # aggregate across seeds
```

Each script prints its full argument list with `--help`.

## Notes

- Rewards are computed on prepared candidates (AST cleaning, longest parseable prefix, per-assertion salvage); see `src/augment/robust_code.py` and `src/reward/stateful.py`.
- Mutation testing uses MutPy 0.6.1, which requires Python 3.9; coverage and FTR run on Python 3.13 as in the paper. The paper reports a cross-interpreter consistency check for this dual-environment setup.
- Naming note: for historical reasons, some identifiers in the code (e.g. the `rkl_lambda` config key) use "RKL"; the quantity actually computed is `KL(teacher || student)`, i.e. the teacher-to-student forward KL as defined in the paper. Config keys are kept unchanged so that the released YAML files remain byte-identical to those used for the paper's experiments.
