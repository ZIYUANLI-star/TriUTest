2. 通用工具与模板（打地基）

读：

src/utils/io.py（读写JSON/建目录）

src/utils/prompts.py（三类提示词模板：生成/修测/诊断）

src/utils/distributed.py（bf16 支持判定）

目的：理解I/O小工具和提示风格，后面看到“模板填充”不陌生。
看完能回答：系统提示+用户提示如何组织成一条可训练的输入。

3. 数据加载与拼批（SFT/GKD共用）

读：

src/data/json_dataset.py

src/data/collator.py

**目的：**从 JSON 到 Dataset，再到 tokenized 批数据是怎么来的。
要点：

load_supervised_dataset：把 prompt 和 chosen/response 对齐成行。

SFTDataCollator：把“prompt + \n + target”拼接，生成 labels（标准自回归监督）。
看完能回答：为什么训练时不用复杂的多段标注也能跑起来。
**小验证：**在脑中或纸上过一遍：一条样本输入 tokenizer 后的 input_ids/labels 的“位移对齐”。

4. 鲁棒增强（训练与奖励都要用到的变体）

读：

src/augment/robust_code.py

目的：理解重命名/注释扰动这些增强怎么做，后面鲁棒奖励要用。
**看完能回答：**模型为何能“对重命名/注释变化不翻车”。
**小验证：**随便拿一小段 Python，手动套 robust_augment 的思路“脑补”一下结果。

5. 沙箱与奖励（RL 的地心引力）

读：

src/reward/sandbox.py（pytest沙箱、变异体执行）

src/reward/rewarders.py（mutant_kill / 速度稳定 / 鲁棒一致性 / 复合奖励）

目的：知道我们如何“真跑测试”、如何算杀死变异体、如何把速度/稳定性/鲁棒变成标量分数。
要点：

run_pytest_with_code：把 CUT 与生成的 tests 写到临时目录，跑 pytest。

run_mutation_kill：对 CUT 做轻量变异→如果测试失败视为击杀。

composite_reward：权重聚合出单一标量（GRPO 用）。
看完能回答：“奖励到底是怎么算的？”、“mutant 分支和 speed 分支为什么能跑出两种风格”。
**小验证：**想一想：如果 CUT 正确而测试很弱，“mutant_kill”分是多少？如果测试很快且稳定，“speed/stability”分会高。

6. 蒸馏损失（GKD 的数理核心）

读：

src/losses/gkd.py

目的：看懂 JSD 与 Reverse-KL 是怎么落地到 logits 上的；只给 chosen-logprob 时的近似做法。
要点：

jsd_loss(student_logits, teacher_logits, alpha)

reverse_kl_on_logits(...) vs reverse_kl_on_chosen(...)
**看完能回答：**为什么说 JSD 更稳、Reverse-KL 更鼓励“覆盖教师分布的峰值”。
**小验证：**能否用一句话解释“为什么 KL(T‖S) 是反向 KL”。

7. SFT 训练主程序（第一阶段）

读：

src/train_sft.py

**目的：**串起 base 模型 → LoRA 包装 → 数据集/Collator → Trainer 的标准监督流程。
要点：

prepare_model_for_kbit_training + load_in_4bit=True：QLoRA 基操。

LoraConfig 与 TrainingArguments 对应哪些 yaml 字段。
**看完能回答：**为什么只训 LoRA 也能把能力“注入”到 3B 基座里。
**小验证：**定位 output_dir 的保存点，确认 SFT 产物将被下一阶段 base_model 消费。

8. GKD 蒸馏主程序（第二阶段）

读：

src/train_distill_gkd.py（重点看 DistillTrainer.compute_loss）

目的：在 SFT 基础上叠加蒸馏项（JSD / Reverse-KL）。
要点：

CE + JSD + RKL 按权重加和。

若只有 teacher_logprobs，替换为 reverse_kl_on_chosen 的路径（代码注释位）。
**看完能回答：**如何在不改数据格式的情况下，把“贴教师的分布信息”注入学生。
**小验证：**在脑中模拟：如果 jsd_lambda 调高、rkl_lambda 调低，训练会更“稳”还是更“贴峰值”。

9. GRPO 训练器（第三阶段 on-policy）

读：

src/rl/grpo_trainer.py（核心）

src/train_grpo.py（入口，把数据与奖励、trainer 串起来）

**目的：**理解 group 采样→算每个候选的奖励→组内均值作 baseline 得优势→梯度更新 + KL 正则 + 在线 JSD 的流程。
要点：

generate_group：同一 prompt 采样多条（group_size）。

composite_reward：从 CUT+tests 真跑拿分。

advantages = reward - group_mean：组相对优势（GRPO 的“G”）。

gkd_online：可选在线 JSD 到 ref_model（教师替身/参考）。
看完能回答：为什么 RL 阶段能把模型“往你想要的风格”推（mutant vs speed），以及KL β的作用。
小验证：对比 grpo_mutant_max.yaml vs grpo_speed_stable.yaml 的温度、max_new_tokens、奖励权重差异，推测行为差异。

10. 评测与推理（闭环）

读：

eval/eval_mutation.py（平均击杀率）

eval/eval_speed.py（速度/稳定性指标）

inference/generate.py（离线推理）

目的：确认两条模型各自强在哪里，如何快速打分与导出示例。
**看完能回答：**如何验证“帕累托两分支”的预期。
**小验证：**想一想：如果把 speed 分支拿去跑 eval_mutation.py，结果会怎样？

11. 脚本层（把所有环节跑通）

读：

scripts/run_sft.sh

scripts/run_distill_gkd.sh

scripts/run_grpo_mutant.sh

scripts/run_grpo_speed.sh

目的：看到一键式命令怎么把前面所有部件串起来。
小验证：对照 0 步的 yaml 路径，确认先后顺序（SFT→GKD→GRPO-mutant 与 GRPO-speed 可并行基于同一 GKD 起点）。