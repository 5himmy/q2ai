#!/usr/bin/env python3
"""
visualize_network.py -- Generate a standalone HTML visualization of KaaN's neural network.

Loads the ActorCritic model from the latest checkpoint (or model_bc.pt fallback)
and produces /home/oky/Documents/vscode/q2ai/network.html with:
  - Architecture diagram (SVG)
  - Weight statistics table
  - Action space visualization
"""

import sys
import os
import glob
import math
import html as html_mod
from datetime import datetime

sys.path.insert(0, '/home/oky/Documents/vscode/q2ai/python')
from model import ActorCritic
from q2ai_env import OBS_DIM

import torch
import numpy as np

# ---------------------------------------------------------------------------
# 1. Find and load the best checkpoint
# ---------------------------------------------------------------------------

CKPT_DIR = '/home/oky/Documents/vscode/q2ai/python/checkpoints'
BC_MODEL = '/home/oky/Documents/vscode/q2ai/python/model_bc.pt'
OUTPUT   = '/home/oky/Documents/vscode/q2ai/network.html'

def find_latest_checkpoint():
    """Return the path to the latest checkpoint, or model_bc.pt as fallback."""
    pattern = os.path.join(CKPT_DIR, 'model_*.pt')
    files = glob.glob(pattern)
    if files:
        # Extract iteration number and pick highest
        def extract_iter(p):
            base = os.path.basename(p)           # model_100.pt
            num = base.replace('model_', '').replace('.pt', '')
            try:
                return int(num)
            except ValueError:
                return -1
        files.sort(key=extract_iter)
        return files[-1]
    if os.path.exists(BC_MODEL):
        return BC_MODEL
    return None

ckpt_path = find_latest_checkpoint()
if ckpt_path is None:
    print("ERROR: No checkpoint found in checkpoints/ and no model_bc.pt")
    sys.exit(1)

print(f"Loading model from: {ckpt_path}")

model = ActorCritic(obs_dim=OBS_DIM)
state = torch.load(ckpt_path, map_location='cpu', weights_only=True)

# Handle both raw state_dict and wrapped checkpoint formats
if isinstance(state, dict) and 'model_state_dict' in state:
    model.load_state_dict(state['model_state_dict'], strict=False)
    ckpt_meta = state
elif isinstance(state, dict) and any(k.startswith('actor') or k.startswith('trunk') for k in state.keys()):
    model.load_state_dict(state, strict=False)
    ckpt_meta = {}
else:
    model.load_state_dict(state, strict=False)
    ckpt_meta = {}

model.eval()
print(f"Model loaded. OBS_DIM={OBS_DIM}, params={sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------------------
# 2. Extract weight statistics
# ---------------------------------------------------------------------------

layer_stats = []
for name, param in model.named_parameters():
    data = param.detach().cpu().numpy()
    layer_stats.append({
        'name': name,
        'shape': str(list(data.shape)),
        'numel': data.size,
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'is_log_std': name == 'action_log_std',
    })

total_params = sum(s['numel'] for s in layer_stats)

# ---------------------------------------------------------------------------
# 3. Extract action head details
# ---------------------------------------------------------------------------

with torch.no_grad():
    actor_bias = model.actor.bias.cpu().numpy()
    actor_weight = model.actor.weight.cpu().numpy()
    log_std = model.action_log_std.cpu().numpy()

# Continuous: indices 0-3, Binary: 4-6, Weapon: 7-17
cont_labels = ['yaw_delta', 'pitch_delta', 'forward', 'side']
binary_labels = ['jump', 'crouch', 'attack']
weapon_labels = [f'weapon_{i}' for i in range(11)]

binary_biases = actor_bias[4:7]
binary_probs = 1.0 / (1.0 + np.exp(-binary_biases))  # sigmoid

log_std_clamped = np.clip(log_std, -2.0, 0.5)
std_vals = np.exp(log_std_clamped)

# ---------------------------------------------------------------------------
# 4. Generate HTML
# ---------------------------------------------------------------------------

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
ckpt_name = os.path.basename(ckpt_path)

def fmt(v, decimals=4):
    """Format a float for display."""
    if abs(v) < 1e-6:
        return "0.0"
    return f"{v:.{decimals}f}"

def color_val(v):
    """Return CSS class for positive/negative."""
    if v > 0.01:
        return 'positive'
    elif v < -0.01:
        return 'negative'
    return ''

# Build the weight stats table rows
stats_rows = ""
for s in layer_stats:
    highlight = ' class="highlight-row"' if s['is_log_std'] else ''
    name_display = s['name']
    if s['is_log_std']:
        name_display += ' <span class="badge">learned exploration</span>'
    stats_rows += f"""<tr{highlight}>
  <td>{name_display}</td>
  <td>{s['shape']}</td>
  <td class="num">{s['numel']:,}</td>
  <td class="num {color_val(s['mean'])}">{fmt(s['mean'])}</td>
  <td class="num">{fmt(s['std'])}</td>
  <td class="num {color_val(s['min'])}">{fmt(s['min'])}</td>
  <td class="num {color_val(s['max'])}">{fmt(s['max'])}</td>
</tr>
"""

# Build log_std rows
log_std_rows = ""
for i, label in enumerate(cont_labels):
    raw = log_std[i]
    clamped = log_std_clamped[i]
    std = std_vals[i]
    log_std_rows += f"""<tr>
  <td>{label}</td>
  <td class="num">{fmt(raw)}</td>
  <td class="num">{fmt(clamped)}</td>
  <td class="num">{fmt(std)}</td>
  <td class="num">{fmt(std * 1.96)}</td>
</tr>
"""

# Build binary action rows
binary_rows = ""
for i, label in enumerate(binary_labels):
    bias = binary_biases[i]
    prob = binary_probs[i]
    pct = prob * 100
    bar_w = max(1, min(200, int(prob * 200)))
    bar_color = '#e74c3c' if prob > 0.5 else '#3498db' if prob > 0.1 else '#555'
    binary_rows += f"""<tr>
  <td>{label}</td>
  <td class="num {color_val(bias)}">{fmt(bias, 2)}</td>
  <td class="num">{fmt(prob, 4)}</td>
  <td class="num">{pct:.2f}%</td>
  <td><div class="prob-bar" style="width:{bar_w}px;background:{bar_color}"></div></td>
</tr>
"""

# Build weapon bias rows
weapon_bias_rows = ""
weapon_biases = actor_bias[7:18]
weapon_softmax = np.exp(weapon_biases - np.max(weapon_biases))
weapon_softmax = weapon_softmax / weapon_softmax.sum()
for i in range(11):
    label = f"weapon {i}" if i > 0 else "no switch"
    bias = weapon_biases[i]
    prob = weapon_softmax[i]
    pct = prob * 100
    bar_w = max(1, min(200, int(prob * 200)))
    bar_color = '#2ecc71' if prob > 0.3 else '#3498db' if prob > 0.1 else '#555'
    weapon_bias_rows += f"""<tr>
  <td>{label}</td>
  <td class="num {color_val(bias)}">{fmt(bias, 3)}</td>
  <td class="num">{pct:.1f}%</td>
  <td><div class="prob-bar" style="width:{bar_w}px;background:{bar_color}"></div></td>
</tr>
"""

# ---------------------------------------------------------------------------
# 5. SVG Architecture Diagram
# ---------------------------------------------------------------------------

SVG_W = 1100
SVG_H = 620

def make_svg():
    """Generate the architecture SVG as a string."""
    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {SVG_H}" '
                 f'width="100%" height="auto" style="max-width:{SVG_W}px">')

    # Defs: gradients, filters
    lines.append("""<defs>
  <linearGradient id="gActor" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#3a7bd5"/>
    <stop offset="100%" stop-color="#2a5db0"/>
  </linearGradient>
  <linearGradient id="gCritic" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#27ae60"/>
    <stop offset="100%" stop-color="#1e8449"/>
  </linearGradient>
  <linearGradient id="gInput" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#8e44ad"/>
    <stop offset="100%" stop-color="#6c3483"/>
  </linearGradient>
  <linearGradient id="gHead" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#e67e22"/>
    <stop offset="100%" stop-color="#d35400"/>
  </linearGradient>
  <filter id="shadow">
    <feDropShadow dx="2" dy="2" stdDeviation="3" flood-color="#000" flood-opacity="0.5"/>
  </filter>
</defs>""")

    # Background
    lines.append(f'<rect width="{SVG_W}" height="{SVG_H}" fill="#0d0d15" rx="10"/>')

    # Title
    lines.append('<text x="550" y="30" text-anchor="middle" fill="#888" '
                 'font-family="Segoe UI,system-ui,sans-serif" font-size="14">'
                 'KaaN Neural Network Architecture</text>')

    # Helper: draw a layer node
    def node(x, y, w, h, label, sublabel, grad_id, dim_text=None):
        r = []
        r.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
                 f'fill="url(#{grad_id})" filter="url(#shadow)" opacity="0.9"/>')
        r.append(f'<text x="{x + w/2}" y="{y + h/2 - 4}" text-anchor="middle" '
                 f'fill="#fff" font-family="Segoe UI,system-ui,sans-serif" font-size="13" '
                 f'font-weight="bold">{label}</text>')
        r.append(f'<text x="{x + w/2}" y="{y + h/2 + 12}" text-anchor="middle" '
                 f'fill="rgba(255,255,255,0.7)" font-family="Segoe UI,system-ui,sans-serif" '
                 f'font-size="11">{sublabel}</text>')
        if dim_text:
            r.append(f'<text x="{x + w/2}" y="{y - 6}" text-anchor="middle" '
                     f'fill="#999" font-family="monospace" font-size="10">{dim_text}</text>')
        return '\n'.join(r)

    # Helper: draw a connection line
    def conn(x1, y1, x2, y2, color='#555'):
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{color}" stroke-width="1.5" opacity="0.6"/>')

    # Helper: draw a bundled connection (multiple lines for "dense" look)
    def dense_conn(x1, y1, x2, y2, color='#555', n=5):
        r = []
        for i in range(n):
            off = (i - n//2) * 3
            r.append(f'<line x1="{x1}" y1="{y1 + off}" x2="{x2}" y2="{y2 + off}" '
                     f'stroke="{color}" stroke-width="0.8" opacity="0.3"/>')
        # Center line solid
        r.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                 f'stroke="{color}" stroke-width="1.5" opacity="0.5"/>')
        return '\n'.join(r)

    # Layout: two networks side by side
    # Input node (shared concept, drawn twice)
    # Actor: top half, Critic: bottom half

    nw = 110  # node width
    nh = 50   # node height

    # ---- ACTOR NETWORK (top) ----
    actor_y_base = 80
    lines.append('<text x="60" y="65" fill="#3a7bd5" font-family="Segoe UI,system-ui,sans-serif" '
                 'font-size="16" font-weight="bold">Actor Network</text>')

    # Input
    ax_in = 40
    lines.append(node(ax_in, actor_y_base, nw, nh, 'Input', f'{OBS_DIM} features', 'gInput', f'dim={OBS_DIM}'))

    # Hidden 1
    ax_h1 = 200
    lines.append(dense_conn(ax_in + nw, actor_y_base + nh//2, ax_h1, actor_y_base + nh//2, '#7D5BA6'))
    lines.append(node(ax_h1, actor_y_base, nw, nh, 'Dense+ReLU', '256 units', 'gActor', 'dim=256'))

    # Hidden 2
    ax_h2 = 360
    lines.append(dense_conn(ax_h1 + nw, actor_y_base + nh//2, ax_h2, actor_y_base + nh//2, '#4A7BD5'))
    lines.append(node(ax_h2, actor_y_base, nw, nh, 'Dense+ReLU', '256 units', 'gActor', 'dim=256'))

    # Hidden 3
    ax_h3 = 520
    lines.append(dense_conn(ax_h2 + nw, actor_y_base + nh//2, ax_h3, actor_y_base + nh//2, '#4A7BD5'))
    lines.append(node(ax_h3, actor_y_base, nw, nh, 'Dense+ReLU', '128 units', 'gActor', 'dim=128'))

    # Actor head
    ax_head = 680
    lines.append(dense_conn(ax_h3 + nw, actor_y_base + nh//2, ax_head, actor_y_base + nh//2, '#4A7BD5'))
    lines.append(node(ax_head, actor_y_base, nw, nh, 'Actor Head', '18 outputs', 'gActor', 'dim=18'))

    # Split into 3 output groups
    # Continuous (4)
    out_x = 840
    out_cont_y = actor_y_base - 70
    out_bin_y = actor_y_base
    out_weap_y = actor_y_base + 70

    lines.append(conn(ax_head + nw, actor_y_base + nh//4, out_x, out_cont_y + nh//2, '#e67e22'))
    lines.append(conn(ax_head + nw, actor_y_base + nh//2, out_x, out_bin_y + nh//2, '#e67e22'))
    lines.append(conn(ax_head + nw, actor_y_base + 3*nh//4, out_x, out_weap_y + nh//2, '#e67e22'))

    lines.append(node(out_x, out_cont_y, 200, nh, 'Continuous (4)', 'yaw, pitch, fwd, side', 'gHead'))
    lines.append(node(out_x, out_bin_y, 200, nh, 'Binary (3)', 'jump, crouch, attack', 'gHead'))
    lines.append(node(out_x, out_weap_y, 200, nh, 'Weapon (11)', 'Categorical select', 'gHead'))

    # Log-std annotation
    log_std_y = out_cont_y - 8
    lines.append(f'<text x="{out_x + 210}" y="{log_std_y + 20}" fill="#e67e22" '
                 f'font-family="monospace" font-size="10">+ action_log_std (learned)</text>')

    # Bernoulli annotation
    lines.append(f'<text x="{out_x + 210}" y="{out_bin_y + 20}" fill="#e67e22" '
                 f'font-family="monospace" font-size="10">Bernoulli(logits)</text>')

    # Categorical annotation
    lines.append(f'<text x="{out_x + 210}" y="{out_weap_y + 20}" fill="#e67e22" '
                 f'font-family="monospace" font-size="10">Categorical(logits)</text>')

    # Divider line
    div_y = 240
    lines.append(f'<line x1="20" y1="{div_y}" x2="{SVG_W - 20}" y2="{div_y}" '
                 f'stroke="#333" stroke-width="1" stroke-dasharray="8,4"/>')

    # ---- CRITIC NETWORK (bottom) ----
    critic_y_base = 290
    lines.append('<text x="60" y="275" fill="#27ae60" font-family="Segoe UI,system-ui,sans-serif" '
                 'font-size="16" font-weight="bold">Critic Network</text>')

    # Input
    cx_in = 40
    lines.append(node(cx_in, critic_y_base, nw, nh, 'Input', f'{OBS_DIM} features', 'gInput', f'dim={OBS_DIM}'))

    # Hidden 1
    cx_h1 = 200
    lines.append(dense_conn(cx_in + nw, critic_y_base + nh//2, cx_h1, critic_y_base + nh//2, '#7D5BA6'))
    lines.append(node(cx_h1, critic_y_base, nw, nh, 'Dense+ReLU', '256 units', 'gCritic', 'dim=256'))

    # Hidden 2
    cx_h2 = 360
    lines.append(dense_conn(cx_h1 + nw, critic_y_base + nh//2, cx_h2, critic_y_base + nh//2, '#27ae60'))
    lines.append(node(cx_h2, critic_y_base, nw, nh, 'Dense+ReLU', '256 units', 'gCritic', 'dim=256'))

    # Hidden 3
    cx_h3 = 520
    lines.append(dense_conn(cx_h2 + nw, critic_y_base + nh//2, cx_h3, critic_y_base + nh//2, '#27ae60'))
    lines.append(node(cx_h3, critic_y_base, nw, nh, 'Dense+ReLU', '128 units', 'gCritic', 'dim=128'))

    # Critic head
    cx_head = 680
    lines.append(dense_conn(cx_h3 + nw, critic_y_base + nh//2, cx_head, critic_y_base + nh//2, '#27ae60'))
    lines.append(node(cx_head, critic_y_base, nw, nh, 'Value Head', '1 scalar', 'gCritic', 'dim=1'))

    # Value output
    vout_x = 840
    lines.append(conn(cx_head + nw, critic_y_base + nh//2, vout_x, critic_y_base + nh//2, '#27ae60'))
    lines.append(node(vout_x, critic_y_base, 160, nh, 'State Value', 'V(s)', 'gCritic'))

    # ---- PARAMETER COUNTS ----
    info_y = 400

    actor_trunk_params = sum(p.numel() for p in model.actor_trunk.parameters())
    actor_head_params = sum(p.numel() for p in model.actor.parameters())
    critic_trunk_params = sum(p.numel() for p in model.critic_trunk.parameters())
    critic_head_params = sum(p.numel() for p in model.critic.parameters())
    log_std_params = model.action_log_std.numel()

    param_info = [
        ('Actor Trunk', actor_trunk_params, '#3a7bd5'),
        ('Actor Head', actor_head_params, '#3a7bd5'),
        ('Critic Trunk', critic_trunk_params, '#27ae60'),
        ('Critic Head', critic_head_params, '#27ae60'),
        ('Log Std', log_std_params, '#e67e22'),
        ('Total', total_params, '#fff'),
    ]

    lines.append(f'<text x="60" y="{info_y}" fill="#888" font-family="Segoe UI,system-ui,sans-serif" '
                 f'font-size="13" font-weight="bold">Parameter Counts</text>')

    for i, (plabel, pcount, pcolor) in enumerate(param_info):
        px = 60 + (i % 3) * 340
        py = info_y + 22 + (i // 3) * 20
        lines.append(f'<text x="{px}" y="{py}" fill="{pcolor}" '
                     f'font-family="monospace" font-size="12">'
                     f'{plabel}: {pcount:,}</text>')

    # ---- LAYER DIMENSION FLOW ----
    flow_y = 480
    lines.append(f'<text x="60" y="{flow_y}" fill="#888" font-family="Segoe UI,system-ui,sans-serif" '
                 f'font-size="13" font-weight="bold">Data Flow (Actor)</text>')

    flow_dims = [
        (f'{OBS_DIM}', 'obs'),
        ('256', 'h1'),
        ('256', 'h2'),
        ('128', 'h3'),
        ('4+3+11=18', 'head'),
    ]

    for i, (dim, lbl) in enumerate(flow_dims):
        fx = 60 + i * 200
        fy = flow_y + 15
        # Arrow
        if i > 0:
            lines.append(f'<text x="{fx - 30}" y="{fy + 14}" fill="#555" '
                         f'font-family="monospace" font-size="16">&rarr;</text>')
        # Dim box
        bw = len(dim) * 10 + 20
        lines.append(f'<rect x="{fx}" y="{fy}" width="{bw}" height="24" rx="4" '
                     f'fill="#1a1a2a" stroke="#444" stroke-width="1"/>')
        lines.append(f'<text x="{fx + bw/2}" y="{fy + 16}" text-anchor="middle" fill="#ddd" '
                     f'font-family="monospace" font-size="12">{dim}</text>')
        lines.append(f'<text x="{fx + bw/2}" y="{fy + 38}" text-anchor="middle" fill="#666" '
                     f'font-family="monospace" font-size="10">{lbl}</text>')

    # Initialization note
    lines.append(f'<text x="60" y="{SVG_H - 20}" fill="#555" '
                 f'font-family="Segoe UI,system-ui,sans-serif" font-size="11">'
                 f'Initialization: orthogonal (std=sqrt(2) trunk, std=0.01 actor head, std=1.0 critic head)</text>')

    lines.append('</svg>')
    return '\n'.join(lines)

svg_content = make_svg()

# ---------------------------------------------------------------------------
# 6. Assemble HTML
# ---------------------------------------------------------------------------

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KaaN Neural Network Visualizer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #0a0a0f;
  color: #e0e0e0;
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}}
h1 {{ color: #ff6b35; margin-bottom: 5px; }}
.subtitle {{ color: #888; margin-bottom: 25px; font-size: 14px; }}
.section {{ margin-bottom: 35px; }}
.section h2 {{
  color: #4ecdc4;
  margin-bottom: 15px;
  border-bottom: 1px solid #333;
  padding-bottom: 5px;
  font-size: 18px;
}}
.section h3 {{
  color: #bbb;
  margin: 15px 0 10px 0;
  font-size: 15px;
}}

/* SVG container */
.svg-container {{
  background: #0d0d15;
  border-radius: 10px;
  padding: 15px;
  border: 1px solid #222;
  overflow-x: auto;
}}

/* Tables */
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin-bottom: 15px;
}}
th {{
  background: #1a1a2a;
  color: #4ecdc4;
  padding: 8px 10px;
  text-align: left;
  font-weight: 600;
  position: sticky;
  top: 0;
}}
td {{
  padding: 6px 10px;
  border-bottom: 1px solid #1a1a2a;
}}
tr:hover {{
  background: #151525;
}}
.num {{
  font-family: 'Cascadia Code', 'Fira Code', monospace;
  text-align: right;
}}
.positive {{ color: #2ecc71; }}
.negative {{ color: #e74c3c; }}

.highlight-row {{
  background: #1a2a1a !important;
}}
.highlight-row:hover {{
  background: #1a3a1a !important;
}}

.badge {{
  display: inline-block;
  background: #e67e22;
  color: #fff;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 3px;
  margin-left: 6px;
  font-weight: normal;
  vertical-align: middle;
}}

/* Probability bars */
.prob-bar {{
  height: 16px;
  border-radius: 3px;
  display: inline-block;
  min-width: 2px;
}}

/* Action cards */
.action-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}}
@media (max-width: 800px) {{
  .action-grid {{
    grid-template-columns: 1fr;
  }}
}}
.action-card {{
  background: #151520;
  border-radius: 8px;
  padding: 15px;
  border: 1px solid #222;
}}
.action-card h3 {{
  margin-top: 0;
}}

/* Stats summary */
.stat-bar {{
  display: flex;
  gap: 15px;
  flex-wrap: wrap;
  margin-bottom: 20px;
}}
.stat-item {{
  background: #151520;
  padding: 12px 18px;
  border-radius: 8px;
  text-align: center;
  border: 1px solid #222;
  min-width: 140px;
}}
.stat-item .val {{
  font-size: 24px;
  font-weight: bold;
  color: #ff6b35;
  font-family: 'Cascadia Code', 'Fira Code', monospace;
}}
.stat-item .lbl {{
  color: #888;
  font-size: 11px;
  margin-top: 4px;
}}

/* Explanation text */
.explain {{
  color: #999;
  font-size: 12px;
  margin-top: 8px;
  line-height: 1.5;
  padding: 8px 12px;
  background: #111118;
  border-radius: 6px;
  border-left: 3px solid #444;
}}
.explain code {{
  color: #e67e22;
  font-family: 'Cascadia Code', 'Fira Code', monospace;
  font-size: 12px;
}}
</style>
</head>
<body>

<h1>KaaN -- Neural Network Visualizer</h1>
<p class="subtitle">Model: {html_mod.escape(ckpt_name)} | Generated: {now} | OBS_DIM: {OBS_DIM}</p>

<div class="stat-bar">
  <div class="stat-item"><div class="val">{total_params:,}</div><div class="lbl">Total Parameters</div></div>
  <div class="stat-item"><div class="val">{OBS_DIM}</div><div class="lbl">Input Features</div></div>
  <div class="stat-item"><div class="val">18</div><div class="lbl">Action Outputs</div></div>
  <div class="stat-item"><div class="val">3</div><div class="lbl">Hidden Layers</div></div>
  <div class="stat-item"><div class="val">2</div><div class="lbl">Networks (A/C)</div></div>
</div>

<!-- Architecture Diagram -->
<div class="section">
<h2>Architecture Diagram</h2>
<div class="svg-container">
{svg_content}
</div>
</div>

<!-- Weight Statistics -->
<div class="section">
<h2>Weight Statistics</h2>
<table>
<tr>
  <th>Layer</th>
  <th>Shape</th>
  <th style="text-align:right">Params</th>
  <th style="text-align:right">Mean</th>
  <th style="text-align:right">Std</th>
  <th style="text-align:right">Min</th>
  <th style="text-align:right">Max</th>
</tr>
{stats_rows}
</table>
</div>

<!-- Action Space -->
<div class="section">
<h2>Action Space Visualization</h2>

<div class="action-grid">

<div class="action-card">
<h3>Continuous Actions -- Exploration (log_std)</h3>
<table>
<tr><th>Action</th><th style="text-align:right">Raw log_std</th><th style="text-align:right">Clamped</th><th style="text-align:right">Std Dev</th><th style="text-align:right">95% Range</th></tr>
{log_std_rows}
</table>
<div class="explain">
<strong>How to read:</strong> The <code>action_log_std</code> parameter controls exploration noise.
<code>std</code> is the standard deviation of the Gaussian around the mean action.
Lower values = more deterministic (exploiting). Higher values = more random (exploring).
Values are clamped to [-2.0, 0.5] before exp().
The 95% range shows the typical deviation from the predicted mean.
</div>
</div>

<div class="action-card">
<h3>Binary Actions -- Initial Bias</h3>
<table>
<tr><th>Action</th><th style="text-align:right">Logit Bias</th><th style="text-align:right">Prob</th><th style="text-align:right">Percent</th><th>Probability Bar</th></tr>
{binary_rows}
</table>
<div class="explain">
<strong>How to read:</strong> These biases set the initial (before training) probability of each binary action.
Negative bias = action rarely taken. <code>sigmoid(bias)</code> gives the probability.
<br>- <code>jump</code> bias of {fmt(binary_biases[0], 1)} = {binary_probs[0]*100:.1f}% initial fire rate
<br>- <code>crouch</code> bias of {fmt(binary_biases[1], 1)} = {binary_probs[1]*100:.4f}% (effectively disabled)
<br>- <code>attack</code> bias of {fmt(binary_biases[2], 1)} = {binary_probs[2]*100:.2f}% (prevents rocket suicide)
</div>
</div>

<div class="action-card">
<h3>Weapon Selection -- Logit Biases</h3>
<table>
<tr><th>Weapon</th><th style="text-align:right">Logit</th><th style="text-align:right">Softmax %</th><th>Distribution</th></tr>
{weapon_bias_rows}
</table>
<div class="explain">
<strong>How to read:</strong> Weapon selection uses a Categorical distribution over 11 options.
<code>softmax(logits)</code> gives the probability of selecting each weapon.
Weapon 0 = "no switch" (keep current weapon). Higher logit = more likely to be selected.
These are the current (possibly trained) logit values.
</div>
</div>

<div class="action-card">
<h3>Continuous Action Head -- Biases</h3>
<table>
<tr><th>Output</th><th style="text-align:right">Bias</th><th>Interpretation</th></tr>
<tr><td>yaw_delta</td><td class="num">{fmt(actor_bias[0], 4)}</td><td>Turn tendency (scaled *10 degrees)</td></tr>
<tr><td>pitch_delta</td><td class="num">{fmt(actor_bias[1], 4)}</td><td>Look up/down tendency (scaled *5 degrees)</td></tr>
<tr><td>forward</td><td class="num">{fmt(actor_bias[2], 4)}</td><td>Forward movement tendency (clipped [-1,1])</td></tr>
<tr><td>side</td><td class="num">{fmt(actor_bias[3], 4)}</td><td>Strafe tendency (clipped [-1,1])</td></tr>
</table>
<div class="explain">
<strong>How to read:</strong> These biases shift the mean output of each continuous action.
A forward bias of {fmt(actor_bias[2], 2)} gives a slight tendency to move forward.
The actual action is sampled from <code>Normal(mean + bias, std)</code> then clipped.
</div>
</div>

</div>
</div>

<!-- Full Actor Output Breakdown -->
<div class="section">
<h2>Actor Head Output Map</h2>
<table>
<tr><th>Index</th><th>Name</th><th style="text-align:right">Weight Bias</th><th>Distribution</th><th>Post-processing</th></tr>
<tr><td>0</td><td>yaw_delta</td><td class="num">{fmt(actor_bias[0])}</td><td>Normal(mean, std)</td><td>* 10.0, clip [-10, 10]</td></tr>
<tr><td>1</td><td>pitch_delta</td><td class="num">{fmt(actor_bias[1])}</td><td>Normal(mean, std)</td><td>* 5.0, clip [-5, 5]</td></tr>
<tr><td>2</td><td>forward</td><td class="num">{fmt(actor_bias[2])}</td><td>Normal(mean, std)</td><td>clip [-1, 1]</td></tr>
<tr><td>3</td><td>side</td><td class="num">{fmt(actor_bias[3])}</td><td>Normal(mean, std)</td><td>clip [-1, 1]</td></tr>
<tr><td>4</td><td>jump</td><td class="num {color_val(actor_bias[4])}">{fmt(actor_bias[4])}</td><td>Bernoulli(logit)</td><td>0 or 1</td></tr>
<tr><td>5</td><td>crouch</td><td class="num {color_val(actor_bias[5])}">{fmt(actor_bias[5])}</td><td>Bernoulli(logit)</td><td>0 or 1</td></tr>
<tr><td>6</td><td>attack</td><td class="num {color_val(actor_bias[6])}">{fmt(actor_bias[6])}</td><td>Bernoulli(logit)</td><td>0 or 1</td></tr>"""

for i in range(11):
    wlabel = f"weapon_{i}" if i > 0 else "no_switch"
    html_content += f"""
<tr><td>{7+i}</td><td>{wlabel}</td><td class="num {color_val(actor_bias[7+i])}">{fmt(actor_bias[7+i])}</td><td>Categorical(logits)</td><td>argmax of 11</td></tr>"""

html_content += """
</table>
</div>

</body>
</html>
"""

# ---------------------------------------------------------------------------
# 7. Write output
# ---------------------------------------------------------------------------

with open(OUTPUT, 'w') as f:
    f.write(html_content)

print(f"Generated: {OUTPUT}")
print(f"Total parameters: {total_params:,}")
print(f"Actor trunk: {sum(p.numel() for p in model.actor_trunk.parameters()):,}")
print(f"Critic trunk: {sum(p.numel() for p in model.critic_trunk.parameters()):,}")
print(f"Log std values: {log_std}")
print(f"Binary biases: jump={actor_bias[4]:.2f} crouch={actor_bias[5]:.2f} attack={actor_bias[6]:.2f}")
print(f"Binary probs:  jump={binary_probs[0]:.4f} crouch={binary_probs[1]:.6f} attack={binary_probs[2]:.4f}")
