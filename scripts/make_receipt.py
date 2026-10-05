"""Render assets/receipt_<model>.png from assets/<model>_param_receipt.json (measured nonzero counts)."""
import json, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
model = sys.argv[1] if len(sys.argv) > 1 else 'csp_yolo1'
g = json.load(open(f'assets/{model}_param_receipt.json'))
tot = g['_total_params']; nz = g['_nonzero']; big = g['bigram_table (dense lookup)']; tr = nz - big; mlp = g['mlp weights']
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
fig.patch.set_facecolor('#0b0d10'); ax.set_facecolor('#0b0d10'); ax.axis('off')
mono = 'DejaVu Sans Mono'
rows = [(f"openai {model}  (the model studied in arXiv 2607.02964)", None, '#9aa4b2'),
        ("parameters in the checkpoint", f"{tot:>13,}", '#e6e6e6'),
        ("  exactly zero", f"-{tot - nz:>12,}", '#5b6470'),
        ("  dense bigram lookup table", f"-{big:>12,}", '#5b6470'),
        ("nonzero transformer parameters", f"{tr:>13,}", '#e6e6e6'),
        ("  of which MLP weights", f"{mlp:>13,}", '#7ee787')]
y = 0.86
for label, val, c in rows:
    ax.text(0.06, y, label, color=c, fontsize=19 if val else 15, family=mono, transform=ax.transAxes)
    if val:
        ax.text(0.94, y, val, color=c, fontsize=19, family=mono, ha='right', transform=ax.transAxes)
    y -= 0.115 if val else 0.13
ax.plot([0.06, 0.94], [y + 0.07, y + 0.07], color='#30363d', lw=1, transform=ax.transAxes)
ax.text(0.06, y - 0.02, f"{tot / tr:.0f}x smaller than it looks.", color='#c9d1d9', fontsize=16, family=mono, transform=ax.transAxes)
fig.savefig(f'assets/receipt_{model}.png', facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.25)
print(f'assets/receipt_{model}.png', round(tot / tr, 1))
