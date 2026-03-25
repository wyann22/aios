import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as mpatches  # type: ignore
import numpy as np

# ── Colors ──
C_NEW    = "#4CAF50"
C_REDUP  = "#F44336"
C_MASK   = "#E0E0E0"
C_READ   = "#2196F3"
C_BG     = "#FAFAFA"
C_OUTPUT = "#FF9800"

LABELS = {"new": "NEW", "dup": "DUP", "mask": "x", "read": "READ"}
COLORS = {"new": C_NEW, "dup": C_REDUP, "mask": C_MASK, "read": C_READ}
TEXT_COLORS = {"new": "white", "dup": "white", "mask": "#999", "read": "white"}

T = ["hello", "my", "name", "is"]
OUT = ["my", "name", "is", "claude"]

FS = 36
S = 1.8
LW = 3.5


def _cell(ax, x, y, val, label=None, w=S, h=S):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=COLORS.get(val, "white"), edgecolor="white", linewidth=LW
    )
    ax.add_patch(rect)
    txt = label if label else LABELS.get(val, "")
    ax.text(x + w / 2, y + h / 2, txt,
            ha="center", va="center", fontsize=FS - 8,
            fontweight="bold", color=TEXT_COLORS.get(val, "black"))


def draw_step(ax, q_states, k_states, attn_matrix, v_states,
              mlp_in, mlp_out, output_token,
              title, q_labels, k_labels, v_labels, mlp_labels):
    """
    Draw one step as a unified flow:

         K (top of attn)        V (vertical, above output)
         k_h  k_m  k_n               V
        [   ][   ][   ]          ┌────────┐
                                 │ v_hello │
    Q       Attn Matrix          │ v_my    │
    q_h  ┌────┬────┬────┐       │ v_name  │
    q_m  │    │    │    │       └────────┘
    q_n  │    │    │    │  softmax     ↓ Σ
         └────┴────┴────┘  ──→   ┌────────┐    ×W     ┌────────┐
             Q · Kᵀ = P          │ a_hello │  ──→     │ o_hello │  → "my"
                                 │ a_my    │          │ o_my    │
                                 │ a_name  │          │ o_name  │
                                 └────────┘          └────────┘
                                   P·V output          MLP out
    """
    n_q = len(q_labels)
    n_k = len(k_labels)
    n_v = len(v_labels)
    n_mlp = len(mlp_labels)
    gap = 0.4 * S
    cw = 1.4 * S  # column cell width

    # ═══ Section 1: Q · Kᵀ ═══
    # Q column on the left
    q_x = 0
    q_base_y = 0
    for i in range(n_q):
        _cell(ax, q_x, q_base_y + (n_q - 1 - i) * S, q_states[i], w=cw)
    for i, lab in enumerate(q_labels):
        ax.text(q_x - 0.12 * S, q_base_y + (n_q - 1 - i) * S + S / 2,
                f"q$_{{{lab}}}$", ha="right", va="center", fontsize=FS - 6)

    # Attn result matrix
    attn_x = q_x + cw + gap
    attn_y = q_base_y
    for i in range(n_q):
        for j in range(n_k):
            _cell(ax, attn_x + j * S, attn_y + (n_q - 1 - i) * S, attn_matrix[i][j])

    # K row on top of attn matrix
    k_y = attn_y + n_q * S + gap * 0.6
    for j in range(n_k):
        _cell(ax, attn_x + j * S, k_y, k_states[j])
    for j, lab in enumerate(k_labels):
        ax.text(attn_x + j * S + S / 2, k_y + S + 0.12 * S,
                f"k$_{{{lab}}}$", ha="center", va="bottom", fontsize=FS - 6)

    # "×" between Q and attn
    ax.text(q_x + cw + gap / 2, attn_y + n_q * S / 2, r"$\times$",
            ha="center", va="center", fontsize=FS - 4, color="#555")

    # Label under attn matrix
    attn_center_x = attn_x + n_k * S / 2
    ax.text(attn_center_x, attn_y - 0.35 * S,
            r"$Q \cdot K^T$  (= P after softmax)",
            ha="center", va="top", fontsize=FS - 8, color="#777", fontstyle="italic")

    # ═══ Arrow from attn to P·V output ═══
    pv_gap = 1.2 * S
    arrow_start_x = attn_x + n_k * S + 0.15 * S
    out_col_x = arrow_start_x + pv_gap
    out_col_w = cw

    # "softmax → Σ" label
    ax.text((arrow_start_x + out_col_x) / 2, attn_y + n_q * S / 2,
            r"$\times V$" + "\n" + r"$\rightarrow$",
            ha="center", va="center", fontsize=FS - 6, color="#555")

    # ═══ Section 2: P·V output column ═══
    for i in range(n_q):
        non_mask = [v for v in attn_matrix[i] if v != "mask"]
        row_status = "dup" if (non_mask and all(v == "dup" for v in non_mask)) else "new"
        _cell(ax, out_col_x, attn_y + (n_q - 1 - i) * S, row_status,
              label=f"a$_{{{q_labels[i]}}}$", w=out_col_w)

    # Label under P·V output
    ax.text(out_col_x + out_col_w / 2, attn_y - 0.35 * S,
            r"$P \cdot V$",
            ha="center", va="top", fontsize=FS - 8, color="#777", fontstyle="italic")

    # ═══ V column above P·V output ═══
    v_center_x = out_col_x + out_col_w / 2
    v_left = v_center_x - cw / 2
    v_base_y = attn_y + n_q * S + gap * 0.6

    for j in range(n_v):
        _cell(ax, v_left, v_base_y + (n_v - 1 - j) * S, v_states[j], w=cw)
    for j, lab in enumerate(v_labels):
        ax.text(v_left - 0.12 * S, v_base_y + (n_v - 1 - j) * S + S / 2,
                f"v$_{{{lab}}}$", ha="right", va="center", fontsize=FS - 6)

    # "V" title above
    ax.text(v_center_x, v_base_y + n_v * S + 0.15 * S,
            "V", ha="center", va="bottom", fontsize=FS - 2, fontweight="bold", color="#555")

    # Dashed arrow from V down to output column
    ax.annotate("", xy=(v_center_x, attn_y + n_q * S + 0.05 * S),
                xytext=(v_center_x, v_base_y - 0.08 * S),
                arrowprops=dict(arrowstyle="->", color="#888", lw=2, linestyle="dashed"))

    # ═══ Section 3: MLP  (input col → ×W → output col) ═══
    mlp_gap = 1.0 * S

    # MLP input column (= P·V output, re-drawn as MLP input)
    mlp_in_x = out_col_x + out_col_w + mlp_gap
    # Arrow from P·V to MLP input
    for i in range(n_q):
        y_c = attn_y + (n_q - 1 - i) * S + S / 2
        ax.annotate("", xy=(mlp_in_x - 0.05 * S, y_c),
                    xytext=(out_col_x + out_col_w + 0.1 * S, y_c),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=2))

    # Draw MLP input column
    for i in range(n_mlp):
        _cell(ax, mlp_in_x, attn_y + (n_mlp - 1 - i) * S, mlp_in[i], w=cw)

    # "×" symbol
    mult_x = mlp_in_x + cw + 0.25 * S
    ax.text(mult_x, attn_y + n_mlp * S / 2,
            r"$\times$",
            ha="center", va="center", fontsize=FS - 2, color="#555")

    # W matrix box (drawn as a square block above the result)
    w_x = mult_x + 0.4 * S
    w_size = max(n_mlp, 1) * S  # make it square-ish, matching input height
    w_y = attn_y + (n_mlp * S - w_size) / 2  # vertically centered
    w_rect = mpatches.FancyBboxPatch(
        (w_x, w_y), w_size, w_size,
        boxstyle="round,pad=0.06",
        facecolor="#78909C", edgecolor="white", linewidth=LW
    )
    ax.add_patch(w_rect)
    ax.text(w_x + w_size / 2, w_y + w_size / 2,
            r"$W_{mlp}$",
            ha="center", va="center", fontsize=FS - 4,
            fontweight="bold", color="white")

    # "=" after W
    eq_x = w_x + w_size + 0.3 * S
    ax.text(eq_x, attn_y + n_mlp * S / 2,
            "=",
            ha="center", va="center", fontsize=FS - 2, color="#555")

    # MLP output column
    mlp_out_x = eq_x + 0.4 * S
    for i in range(n_mlp):
        _cell(ax, mlp_out_x, attn_y + (n_mlp - 1 - i) * S, mlp_out[i], w=cw)
    for i, lab in enumerate(mlp_labels):
        ax.text(mlp_out_x + cw + 0.12 * S, attn_y + (n_mlp - 1 - i) * S + S / 2,
                lab, ha="left", va="center", fontsize=FS - 8, color="#555")

    # Label under MLP block
    mlp_center = (mlp_in_x + mlp_out_x + cw) / 2
    ax.text(mlp_center, attn_y - 0.35 * S,
            "MLP",
            ha="center", va="top", fontsize=FS - 6, color="#777",
            fontweight="bold", fontstyle="italic")

    # ═══ Section 4: Output token ═══
    out_gap = 0.8 * S
    tok_x = mlp_out_x + cw + out_gap
    tok_w = 2.4 * S
    tok_y = attn_y + n_mlp * S / 2 - S / 2  # vertically centered

    # Arrow
    ax.annotate("", xy=(tok_x - 0.05 * S, tok_y + S / 2),
                xytext=(mlp_out_x + cw + 0.1 * S, tok_y + S / 2),
                arrowprops=dict(arrowstyle="->", color="#333", lw=2.5))

    # Output box
    rect = mpatches.FancyBboxPatch(
        (tok_x, tok_y), tok_w, S,
        boxstyle="round,pad=0.08",
        facecolor=C_OUTPUT, edgecolor="white", linewidth=LW
    )
    ax.add_patch(rect)
    ax.text(tok_x + tok_w / 2, tok_y + S / 2, f'"{output_token}"',
            ha="center", va="center", fontsize=FS,
            fontweight="bold", color="white")

    # ═══ Limits ═══
    max_top = max(k_y + S, v_base_y + n_v * S)
    right_edge = tok_x + tok_w
    ax.set_xlim(-2.0 * S, right_edge + 0.5 * S)
    ax.set_ylim(attn_y - 0.7 * S, max_top + 0.8 * S)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=FS + 2, fontweight="bold", pad=16)
    ax.axis("off")


# ═══════════════════════════════════════════════════════════════
#  Figure 1: WITHOUT KV Cache
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(4, 1, figsize=(50, 72),
                         gridspec_kw={"height_ratios": [1, 1, 1, 1]})
fig.patch.set_facecolor(C_BG)
fig.suptitle('Without KV Cache  —  "hello my name is" auto-regressive generation',
             fontsize=FS + 16, fontweight="bold", y=0.995)

# Step 1
draw_step(axes[0],
          q_states=["new"], k_states=["new"],
          attn_matrix=[["new"]],
          v_states=["new"],
          mlp_in=["new"], mlp_out=["new"], output_token=OUT[0],
          title="Step 1 (prefill)  —  input: [hello]",
          q_labels=[T[0]], k_labels=[T[0]], v_labels=[T[0]], mlp_labels=[T[0]])

# Step 2
draw_step(axes[1],
          q_states=["dup", "new"], k_states=["dup", "new"],
          attn_matrix=[["dup", "mask"],
                       ["new", "new"]],
          v_states=["dup", "new"],
          mlp_in=["dup", "new"], mlp_out=["dup", "new"], output_token=OUT[1],
          title="Step 2 (decode)  —  input: [hello, my]",
          q_labels=T[:2], k_labels=T[:2], v_labels=T[:2], mlp_labels=T[:2])

# Step 3
draw_step(axes[2],
          q_states=["dup", "dup", "new"], k_states=["dup", "dup", "new"],
          attn_matrix=[["dup", "mask", "mask"],
                       ["dup", "dup",  "mask"],
                       ["new", "new",  "new"]],
          v_states=["dup", "dup", "new"],
          mlp_in=["dup", "dup", "new"], mlp_out=["dup", "dup", "new"],
          output_token=OUT[2],
          title="Step 3 (decode)  —  input: [hello, my, name]",
          q_labels=T[:3], k_labels=T[:3], v_labels=T[:3], mlp_labels=T[:3])

# Step 4
draw_step(axes[3],
          q_states=["dup", "dup", "dup", "new"],
          k_states=["dup", "dup", "dup", "new"],
          attn_matrix=[["dup",  "mask", "mask", "mask"],
                       ["dup",  "dup",  "mask", "mask"],
                       ["dup",  "dup",  "dup",  "mask"],
                       ["new",  "new",  "new",  "new"]],
          v_states=["dup", "dup", "dup", "new"],
          mlp_in=["dup", "dup", "dup", "new"],
          mlp_out=["dup", "dup", "dup", "new"],
          output_token=OUT[3],
          title="Step 4 (decode)  —  input: [hello, my, name, is]",
          q_labels=T, k_labels=T, v_labels=T, mlp_labels=T)

legend_elements = [
    mpatches.Patch(facecolor=C_NEW,    edgecolor="gray", label="NEW  — necessary new computation"),
    mpatches.Patch(facecolor=C_REDUP,  edgecolor="gray", label="DUP  — redundant (same result as earlier step)"),
    mpatches.Patch(facecolor=C_MASK,   edgecolor="gray", label="0 / x — causal mask (zeroed out)"),
    mpatches.Patch(facecolor=C_OUTPUT, edgecolor="gray", label="Output — generated token"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=FS + 2,
           frameon=True, fancybox=True, shadow=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig("./no_kv_cache.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved: no_kv_cache.png")


# ═══════════════════════════════════════════════════════════════
#  Figure 2: WITH KV Cache
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(4, 1, figsize=(50, 72),
                         gridspec_kw={"height_ratios": [1, 1, 1, 1]})
fig.patch.set_facecolor(C_BG)
fig.suptitle('With KV Cache  —  "hello my name is" auto-regressive generation',
             fontsize=FS + 16, fontweight="bold", y=0.995)

# Prefill
draw_step(axes[0],
          q_states=["new"], k_states=["new"],
          attn_matrix=[["new"]],
          v_states=["new"],
          mlp_in=["new"], mlp_out=["new"], output_token=OUT[0],
          title="Prefill  —  input: [hello]",
          q_labels=[T[0]], k_labels=[T[0]], v_labels=[T[0]], mlp_labels=[T[0]])

# Decode 1
draw_step(axes[1],
          q_states=["new"], k_states=["read", "new"],
          attn_matrix=[["new", "new"]],
          v_states=["read", "new"],
          mlp_in=["new"], mlp_out=["new"], output_token=OUT[1],
          title="Decode 1  —  input: [my]",
          q_labels=[T[1]], k_labels=T[:2], v_labels=T[:2], mlp_labels=[T[1]])

# Decode 2
draw_step(axes[2],
          q_states=["new"], k_states=["read", "read", "new"],
          attn_matrix=[["new", "new", "new"]],
          v_states=["read", "read", "new"],
          mlp_in=["new"], mlp_out=["new"], output_token=OUT[2],
          title="Decode 2  —  input: [name]",
          q_labels=[T[2]], k_labels=T[:3], v_labels=T[:3], mlp_labels=[T[2]])

# Decode 3
draw_step(axes[3],
          q_states=["new"], k_states=["read", "read", "read", "new"],
          attn_matrix=[["new", "new", "new", "new"]],
          v_states=["read", "read", "read", "new"],
          mlp_in=["new"], mlp_out=["new"], output_token=OUT[3],
          title="Decode 3  —  input: [is]",
          q_labels=[T[3]], k_labels=T, v_labels=T, mlp_labels=[T[3]])

legend_elements = [
    mpatches.Patch(facecolor=C_NEW,    edgecolor="gray", label="NEW   — new computation"),
    mpatches.Patch(facecolor=C_READ,   edgecolor="gray", label="READ  — read K/V from cache (no re-projection)"),
    mpatches.Patch(facecolor=C_OUTPUT, edgecolor="gray", label="Output — generated token"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=FS + 2,
           frameon=True, fancybox=True, shadow=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig("./with_kv_cache.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved: with_kv_cache.png")


# ═══════════════════════════════════════════════════════════════
#  Figure 3: FLOPs comparison (unchanged)
# ═══════════════════════════════════════════════════════════════

FS_BAR = 20
fig, ax = plt.subplots(1, 1, figsize=(20, 9))
fig.patch.set_facecolor(C_BG)

categories = ["Q Proj", "K Proj", "V Proj", r"$Q \cdot K^T$", r"$P \cdot V$", "MLP"]
no_cache   = [10, 10, 10, 10, 10, 10]
with_cache = [4,  4,  4,  10, 4,  4]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, no_cache, width, label="Without KV Cache",
               color=C_REDUP, edgecolor="white", linewidth=2, alpha=0.9)
bars2 = ax.bar(x + width/2, with_cache, width, label="With KV Cache",
               color=C_NEW, edgecolor="white", linewidth=2, alpha=0.9)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            str(int(bar.get_height())), ha="center", fontsize=FS_BAR + 2, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            str(int(bar.get_height())), ha="center", fontsize=FS_BAR + 2, fontweight="bold")

for i in range(len(categories)):
    saved = no_cache[i] - with_cache[i]
    if saved > 0:
        pct = saved / no_cache[i] * 100
        ax.annotate(f"-{pct:.0f}%", xy=(x[i] + width/2, with_cache[i] + 0.8),
                    fontsize=FS_BAR - 2, ha="center", color=C_NEW, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=FS_BAR + 2)
ax.set_ylabel("Token-level operations (sum of 4 steps)", fontsize=FS_BAR + 2)
ax.set_title('Total Compute:  "hello my name is" — 4 auto-regressive steps',
             fontsize=FS_BAR + 10, fontweight="bold")
ax.legend(fontsize=FS_BAR + 2, loc="upper right")
ax.set_ylim(0, 13)
ax.tick_params(axis='y', labelsize=FS_BAR)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("./flops_comparison", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: flops_comparison.png")
