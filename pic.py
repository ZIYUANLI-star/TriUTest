import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

def panel(x0, y0, w, h, title):
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=1.5))
    ax.text(x0 + w/2, y0 + h - 0.05, title,
            ha='center', va='top', fontsize=10, fontweight='bold')

def box(x0, y0, w, h, text, fontsize=9):
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=1.2))
    ax.text(x0 + w/2, y0 + h/2, text,
            ha='center', va='center', fontsize=fontsize)

def arrow(x0, y0, x1, y1, text=None, fontsize=8):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', linewidth=1))
    if text is not None:
        ax.text((x0 + x1)/2, (y0 + y1)/2 + 0.02, text,
                ha='center', va='bottom', fontsize=fontsize)

w_panel, h_panel, y_panel = 0.28, 0.9, 0.05
x_sft, x_gkd, x_grpo = 0.04, 0.36, 0.68

# Panels
panel(x_sft,  y_panel, w_panel, h_panel, "Stage 1: SFT (Supervised Fine-Tuning)")
panel(x_gkd,  y_panel, w_panel, h_panel, "Stage 2: GKD (On-Policy Guided Distillation)")
panel(x_grpo, y_panel, w_panel, h_panel, "Stage 3: GRPO (RL from Execution Feedback)")

# SFT
box(x_sft + 0.02, 0.62, 0.11, 0.18, "Prompt–\nTest\nCorpus")
box(x_sft + 0.15, 0.62, 0.16, 0.18, "Student LLM\n(Qwen2.5-3B)")
box(x_sft + 0.12, 0.30, 0.20, 0.18, r"$\mathcal{M}_{\mathrm{SFT}}$")
arrow(x_sft + 0.13, 0.71, x_sft + 0.15, 0.71,
      "SFT loss\n(CE + semantic reg.)", fontsize=7)
arrow(x_sft + 0.23, 0.62, x_sft + 0.22, 0.48)
ax.text(x_sft + 0.02, 0.18,
        "Goal: learn executable\nPython tests with\nbasic assertion habits.",
        fontsize=8, ha='left', va='top')

# GKD
box(x_gkd + 0.02, 0.62, 0.16, 0.18, r"Student policy" "\n" r"$\mathcal{M}_{\mathrm{SFT}}$")
box(x_gkd + 0.20, 0.62, 0.16, 0.18, "Teacher LLM\n(Qwen2.5-14B)")
arrow(x_gkd + 0.10, 0.62, x_gkd + 0.10, 0.50,
      "sample\ncandidates", fontsize=7)
arrow(x_gkd + 0.28, 0.62, x_gkd + 0.28, 0.50,
      "teacher\nlogits", fontsize=7)
box(x_gkd + 0.06, 0.36, 0.24, 0.18,
    "Masked distillation on\nTest-sensitive fragments\n(CE + JSD + reverse KL)",
    fontsize=8)
arrow(x_gkd + 0.18, 0.50, x_gkd + 0.18, 0.36)
box(x_gkd + 0.09, 0.18, 0.18, 0.14, r"$\mathcal{M}_{\mathrm{GKD}}$")
ax.text(x_gkd + 0.03, 0.12,
        "Mask focuses on\nassertions & inputs;\nkeeps structure flexible.",
        fontsize=8, ha='left', va='top')

# GRPO
box(x_grpo + 0.09, 0.64, 0.18, 0.16, r"Policy model" "\n" r"$\mathcal{M}_{\mathrm{GKD}}$")
arrow(x_grpo + 0.18, 0.64, x_grpo + 0.18, 0.52,
      "sample\nK tests", fontsize=7)
box(x_grpo + 0.06, 0.40, 0.24, 0.18,
    "Execution engine:\n• Correct impl.\n• Mutants per file\n• Buggy impl.",
    fontsize=8)
arrow(x_grpo + 0.18, 0.40, x_grpo + 0.18, 0.32,
      "per-test reward\n$f_{mut}, f_{stab}, f_{ass}$", fontsize=7)
box(x_grpo + 0.04, 0.22, 0.12, 0.12, "Group-wise\nadvantage\n(GRPO)", fontsize=8)
box(x_grpo + 0.20, 0.22, 0.16, 0.12, "Updated policy\nTriUTest", fontsize=8)
arrow(x_grpo + 0.10, 0.22, x_grpo + 0.20, 0.22,
      "update\nLoRA / head", fontsize=7)
ax.text(x_grpo + 0.03, 0.12,
        "Optimizes mutation kill\n& file-level FTR; KL keeps\npolicy near reference.",
        fontsize=8, ha='left', va='top')

# Cross-stage arrows
arrow(x_sft + w_panel, 0.50, x_gkd, 0.50,
      text="student init.", fontsize=8)
arrow(x_gkd + w_panel, 0.50, x_grpo, 0.50,
      text="calibrated policy", fontsize=8)

ax.text(0.5, 0.96,
        "TriUTest: Three-Stage Post-Training for Fault-Oriented Test Generation",
        ha='center', va='top', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("triutest_overview_new.png", dpi=300, bbox_inches='tight')
plt.show()
