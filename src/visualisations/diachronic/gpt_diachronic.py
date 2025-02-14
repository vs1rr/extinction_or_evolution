import matplotlib.pyplot as plt
import numpy as np

# Data for the results
categories = [
    "JAC", "PREC", "REC", "F1",
    "COS", "SEM", "RBO",
    "MPC", "XPC", "KTAU"
]

gpt_adjs_open_high = [0.1318, 0.6665, 0.1411, 0.2329, 0.5137, 0.9977, 0.1608, 0.2142, 1.0000, 0.6332]
gpt_adjs_pay_high = [0.1318, 0.6665, 0.1411, 0.2329, 0.5137, 0.9977, 0.1608, 0.2142, 1.0000, 0.6332]

gpt_adjs_open_low = [0.1318, 0.6665, 0.1411, 0.2329, 0.5137, 0.9977, 0.1608, 0.2142, 1.0000, 0.6332]
gpt_adjs_pay_low = [0.1318, 0.6665, 0.1411, 0.2329, 0.5137, 0.9977, 0.1608, 0.2142, 1.0000, 0.6332]

gpt_advs_open_high = [0.0783, 0.4956, 0.0851, 0.1453, 0.4808, 0.9912, 0.0050, 0.1826, 1.0000, 0.6204]
gpt_advs_pay_high = [0.0783, 0.4956, 0.0851, 0.1453, 0.4808, 0.9912, 0.0050, 0.1826, 1.0000, 0.6204]

gpt_advs_open_low = [0.0783, 0.4956, 0.0851, 0.1453, 0.4808, 0.9912, 0.0050, 0.1826, 1.0000, 0.6204]
gpt_advs_pay_low = [0.0783, 0.4956, 0.0851, 0.1453, 0.4808, 0.9912, 0.0050, 0.1826, 1.0000, 0.6204]

gpt_nouns_open_high = [0.1802, 0.2738, 0.3452, 0.3054, 0.3565, 0.9943, 0.0002, 0.2170, 1.0000, 0.6437]
gpt_nouns_pay_high = [0.1802, 0.2738, 0.3452, 0.3054, 0.3565, 0.9943, 0.0002, 0.2170, 1.0000, 0.6437]

gpt_nouns_open_low = [0.1802, 0.2738, 0.3452, 0.3054, 0.3565, 0.9943, 0.0002, 0.2170, 1.0000, 0.6437]
gpt_nouns_pay_low = [0.1802, 0.2738, 0.3452, 0.3054, 0.3565, 0.9943, 0.0002, 0.2170, 1.0000, 0.6437]

gpt_verbs_open_high = [0.1331, 0.1823, 0.3302, 0.2349, 0.1019, 0.9803, 0.0283, 0.1907, 1.0000, 0.6664]
gpt_verbs_pay_high = [0.1802, 0.2738, 0.3452, 0.3054, 0.3565, 0.9943, 0.0002, 0.2170, 1.0000, 0.6437]

gpt_verbs_open_low = [0.1331, 0.1823, 0.3302, 0.2349, 0.1019, 0.9803, 0.0283, 0.1907, 1.0000, 0.6664]
gpt_verbs_pay_low = [0.1331, 0.1823, 0.3302, 0.2349, 0.1019, 0.9803, 0.0283, 0.1907, 1.0000, 0.6664]

x = np.arange(len(categories))  # Label locations
width = 0.2  # Width of bars

# Define a scientific color palette
colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2","#4e79a7", "#f28e2c", "#e15759", "#76b7b2","#4e79a7", "#f28e2c", "#e15759", "#76b7b2","#76b7b2","#4e79a7", "#f28e2c","#4e79a7"]

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(x - 1.5 * width, gpt_adjs_open_high, width, label="GPT-4 + ADJ_HGH + WordsAPI", color=colors[0])
ax.bar(x - 0.5 * width, gpt_adjs_pay_high, width, label="GPT-4 + ADJ_HGH + INRIAE ", color=colors[1])

ax.bar(x - 0.5 * width, gpt_adjs_open_low, width, label="GPT-4 + ADJ_LOW + WordsAPIW ", color=colors[2])
ax.bar(x - 0.5 * width, gpt_adjs_pay_low, width, label="GPT-4 + ADJ_LOW + INRIAE ", color=colors[3])

ax.bar(x + 0.5 * width, gpt_advs_open_high, width, label="GPT-4 + ADV_HIGH + WordsAPI ", color=colors[4])
ax.bar(x + 0.5 * width, gpt_advs_pay_high, width, label="GPT-4 + ADV_HIGH + INRIAE ", color=colors[5])

ax.bar(x + 1.5 * width, gpt_advs_open_low, width, label="GPT-4 + ADV_LOW + WordsAPI", color=colors[6])
ax.bar(x + 1.5 * width, gpt_advs_pay_low, width, label="GPT-4 + ADV_LOW + INRIAE", color=colors[7])

ax.bar(x + 0.5 * width, gpt_nouns_open_high, width, label="GPT-4 + NOUNS_HIGH + WordsAPI ", color=colors[8])
ax.bar(x + 0.5 * width, gpt_nouns_pay_high, width, label="GPT-4 + NOUNS_HIGH + INRIAE ", color=colors[9])

ax.bar(x + 1.5 * width, gpt_nouns_open_low, width, label="GPT-4 + NOUNS_LOW + WordsAPI", color=colors[10])
ax.bar(x + 1.5 * width, gpt_nouns_pay_low, width, label="GPT-4 + NOUNS_LOW + INRIAE", color=colors[11])

ax.bar(x + 0.5 * width, gpt_verbs_open_high, width, label="GPT-4 + VERBS_HIGH + WordsAPI ", color=colors[12])
ax.bar(x + 0.5 * width, gpt_verbs_pay_high, width, label="GPT-4 + VERBS_HIGH + INRIAE ", color=colors[13])

ax.bar(x + 1.5 * width, gpt_verbs_open_low, width, label="GPT-4 + VERBS_LOW + WordsAPI", color=colors[14])
ax.bar(x + 1.5 * width, gpt_verbs_pay_low, width, label="GPT-4 + VERBS_LOW + INRIAE", color=colors[15])

# Labels, title, and legend
ax.set_ylabel("Scores", fontsize=13)
ax.set_title("GPT-4 Synonyms Generation Comparison for XIXth Centur Gardening Manuals", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=25, ha="right", fontsize=12)  # Reduced rotation angle
ax.legend(fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

