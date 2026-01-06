import matplotlib.pyplot as plt

models = ['Zero-shot 3B', 'SFT only', 'SFT+GKD', 'SFT+GRPO', 'TriUTest']
# 假设的缺陷检出率数据
bug_trigger_rate = [0.15, 0.25, 0.30, 0.32, 0.40]

plt.figure(figsize=(6,4))
plt.bar(models, bug_trigger_rate, color=['gray','skyblue','orange','green','purple'])
plt.ylim(0, 0.5)
plt.ylabel('缺陷检出率')
plt.title('不同方法的缺陷检出率对比（越高越好）')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()