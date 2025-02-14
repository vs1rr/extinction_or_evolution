import matplotlib.pyplot as plt
import numpy as np

# Data for the results
categories = [
    "JAC", "PREC", "REC", "F1",
    "COS", "SEM", "RBO",
    "MPC", "XPC", "KTAU"
]

gpt_simple_mw = [0.1109, 0.8648, 0.1129, 0.1997, 0.6389, 0.9970, 0.1765, 0.2175, 1.0000, 0.6458]
gpt_complex_mw = [0.0589, 0.7304, 0.0602, 0.1113, 0.4867, 0.9807, 0.3192, 0.1876, 1.0000, 0.6389]
gpt_simple_api = [0.2211, 0.4139, 0.3218, 0.3621, 0.4214, 0.9932, 0.0849, 0.2203, 1.0000, 0.7098]
gpt_complex_api = [0.1840, 0.3341, 0.2906, 0.3108, 0.0515, 0.9766, 0.0202, 0.1973, 1.0000, 0.6863]

x = np.arange(len(categories))  # Label locations
width = 0.2  # Width of bars

# Define a scientific color palette
colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2"]

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(x - 1.5 * width, gpt_simple_mw, width, label="GPT-4 + SW + MW", color=colors[0])
ax.bar(x - 0.5 * width, gpt_complex_mw, width, label="GPT-4 + CW + MW ", color=colors[1])
ax.bar(x + 0.5 * width, gpt_simple_api, width, label="GPT-4 + SW + WordsAPI ", color=colors[2])
ax.bar(x + 1.5 * width, gpt_complex_api, width, label="GPT-4 + CW + WordsAPI", color=colors[3])

# Labels, title, and legend
ax.set_ylabel("Scores", fontsize=13)
ax.set_title("GPT Synonyms Generation Comparison for Simple and Complex Words", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=25, ha="right", fontsize=12)  # Reduced rotation angle
ax.legend(fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
