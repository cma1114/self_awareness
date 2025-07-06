import re, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

# ---------- 1. PARSE THE MARKDOWN DUMP ----------
md_path = Path("./sc_dump.md")  
raw = md_path.read_text()

blocks = re.split(r"--- Analyzing ", raw)[1:]        # skip prologue
records = []

def grab(block, var, model_tag):
    """
    Return (coef, p-value) for `var` inside the model section labelled `model_tag`.
    """
    sect = re.search(fr"Model {model_tag}:(.*?)\n\n", block, flags=re.S)
    if not sect:
        return (None, None)
    txt = sect.group(1)
    m_coef = re.search(fr"\n{var}\s+([-\d\.Ee]+)", txt)
    m_pval = re.search(fr"\n{var}\s+[-\d\.Ee]+\s+[-\d\.Ee]+\s+[-\d\.Ee]+\s+([0-9\.Ee-]+)", txt)
    return (float(m_coef.group(1)) if m_coef else None,
            float(m_pval.group(1)) if m_pval else None)

def pseudo(block, tag):
    m = re.search(fr"Model {tag}:.*?Pseudo R-squ\.\s*:\s*([0-9\.]+)", block, flags=re.S)
    return float(m.group(1)) if m else None

for blk in blocks:
    header = blk.splitlines()[0]
    model_id  = re.match(r"([^\s]+)", header).group(1)
    family    = model_id.split('-')[0]        # gpt-4o, gpt-4, gemini, grok …
    correctness = "Correct" if "Correct" in header else "Incorrect"
    dataset   = "GPQA" if "_GPQA_" in blk else ("SimpleMC" if "_SimpleMC_" in blk else "other")

    ent_c , ent_p  = grab(blk, "capabilities_entropy", "1\\.5")
    p1_c  , p1_p   = grab(blk, "p_i_capability",      "1\\.4")
    z1_c  , z1_p   = grab(blk, "p1_z",                "1\\.51")
    z2_c  , z2_p   = grab(blk, "I\\(p1_z \\*\\* 2\\)", "1\\.51")
    records.append({
        "family": family, "run": model_id, "dataset": dataset, "correctness": correctness,
        "entropy_coef": ent_c,  "entropy_p": ent_p,
        "p1_coef": p1_c,        "p1_p": p1_p,
        "p1z_coef": z1_c,       "p1z_p": z1_p,
        "p1z2_coef": z2_c,      "p1z2_p": z2_p,
        "pseudo_1.4": pseudo(blk, "1\\.4"),
        "pseudo_1.5": pseudo(blk, "1\\.5"),
        "pseudo_1.51": pseudo(blk, "1\\.51"),
    })

df = pd.DataFrame(records)
df.to_csv("./second_chance_summary.csv", index=False)   # master table

# ---------- 2. VISUALISATIONS ----------
sns.set_theme(style="whitegrid")

# (a) Entropy coefficients
plt.figure(figsize=(8,4))
sns.barplot(data=df, x="family", y="entropy_coef", hue="correctness", ci=None)
plt.axhline(0, color="black", linewidth=.8)
plt.ylabel("Coefficient of capabilities_entropy  (Model 1.5)")
plt.xlabel("")
plt.title("Entropy consistently raises flip probability")
plt.tight_layout()
plt.savefig("./entropy_coeff_bar.png", dpi=300)
plt.close()

# (b) Peak of quadratic confidence curve
peak_df = df.dropna(subset=["p1z_coef", "p1z2_coef"]).copy()
peak_df["peak_z"] = -peak_df["p1z_coef"] / (2*peak_df["p1z2_coef"])
plt.figure(figsize=(8,4))
sns.stripplot(data=peak_df, x="family", y="peak_z", hue="correctness",
              jitter=True, size=7, alpha=.8)
plt.axhline(0, color="grey", linestyle="--", linewidth=.7)
plt.ylabel("Z-score of p₁ where flips peak")
plt.xlabel("")
plt.title("Flip-rate peaks at moderate confidence (inverted-U)")
plt.tight_layout()
plt.savefig("./peak_location_strip.png", dpi=300)
plt.close()

# (c) Explanatory power across models
pseudo_long = df.melt(id_vars=["run"], value_vars=["pseudo_1.4","pseudo_1.5","pseudo_1.51"],
                      var_name="spec", value_name="pseudoR")
plt.figure(figsize=(7,4))
sns.boxplot(data=pseudo_long, x="spec", y="pseudoR", color="lightgrey")
sns.stripplot(data=pseudo_long, x="spec", y="pseudoR", color="black", size=4, jitter=True, alpha=.6)
plt.ylabel("Pseudo R²")
plt.xlabel("")
plt.title("How much deviance each specification captures")
plt.tight_layout()
plt.savefig("./pseudoR_comparison.png", dpi=300)
plt.close()

print("✅   Summary CSV and three figures written:")
print("    • second_chance_summary.csv")
print("    • entropy_coeff_bar.png")
print("    • peak_location_strip.png")
print("    • pseudoR_comparison.png")
