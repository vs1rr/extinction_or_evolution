import matplotlib.pyplot as plt
import numpy as np

# Data for the results
categories = [
    "JAC", "PREC", "REC", "F1",
    "COS", "SEM", "RBO",
    "MPC", "XPC", "KTAU"
]

llama_simple_open = [0.1318, 0.6665, 0.1411, 0.2329, 0.5137, 0.9977, 0.1608, 0.2142, 1.0000, 0.6332]
llama_complex_open = [0.0783, 0.4956, 0.0851, 0.1453, 0.4808, 0.9912, 0.0050, 0.1826, 1.0000, 0.6204]
llama_simple_pay = [0.1802, 0.2738, 0.3452, 0.3054, 0.3565, 0.9943, 0.0002, 0.2170, 1.0000, 0.6437]
llama_complex_pay = [0.1331, 0.1823, 0.3302, 0.2349, 0.1019, 0.9803, 0.0283, 0.1907, 1.0000, 0.6664]

x = np.arange(len(categories))  # Label locations
width = 0.2  # Width of bars

# Define a scientific color palette
colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2"]

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(x - 1.5 * width, llama_simple_open, width, label="LLAMA + SW + MW", color=colors[0])
ax.bar(x - 0.5 * width, llama_complex_open, width, label="LLAMA + CW + MW ", color=colors[1])
ax.bar(x + 0.5 * width, llama_simple_pay, width, label="LLAMA + SW + WordsAPI ", color=colors[2])
ax.bar(x + 1.5 * width, llama_complex_pay, width, label="LLAMA + CW + WordsAPI", color=colors[3])

# Labels, title, and legend
ax.set_ylabel("Scores", fontsize=13)
ax.set_title("LLAMA Synonyms Generation Comparison for Simple and Complex Words", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=25, ha="right", fontsize=12)  # Reduced rotation angle
ax.legend(fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

