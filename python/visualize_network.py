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
state = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Extract state_dict from wrapped checkpoint formats
if isinstance(state, dict) and 'model' in state:
    state_dict = state['model']
    ckpt_meta = state
elif isinstance(state, dict) and 'model_state_dict' in state:
    state_dict = state['model_state_dict']
    ckpt_meta = state
elif isinstance(state, dict) and any(k.startswith('actor') or k.startswith('trunk') or k.startswith('self_encoder') for k in state.keys()):
    state_dict = state
    ckpt_meta = {}
else:
    state_dict = state
    ckpt_meta = {}

# Migrate old checkpoint formats — skip incompatible weights
keys_to_skip = set()
for k, v in state_dict.items():
    # Old trunk/encoder weights can't map to grouped encoders
    if k.startswith('trunk.') or k.startswith('actor_trunk.') or k.startswith('critic_trunk.'):
        keys_to_skip.add(k)
    elif k.startswith('actor_encoder.') or k.startswith('critic_encoder.'):
        keys_to_skip.add(k)
    # Old heads with wrong input dim (128 vs 256)
    elif k in ('actor.weight', 'critic.weight') and v.shape[1] != 256:
        keys_to_skip.add(k)

if keys_to_skip:
    print(f"  Skipping {len(keys_to_skip)} incompatible keys from old checkpoint")
    state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_skip}

model.load_state_dict(state_dict, strict=False)
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

# Per-dimension clamping: yaw/pitch tighter for precise aim
log_std_clamped = log_std.copy()
log_std_clamped[0] = np.clip(log_std[0], -4.0, -2.5)  # yaw
log_std_clamped[1] = np.clip(log_std[1], -4.0, -2.5)  # pitch
log_std_clamped[2] = np.clip(log_std[2], -2.0, 0.5)   # forward
log_std_clamped[3] = np.clip(log_std[3], -2.0, 0.5)   # side
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
# 5. SVG Architecture Diagram — V3.0 Grouped Encoder
# ---------------------------------------------------------------------------

SVG_W = 1200
SVG_H = 780

def make_svg():
    """Generate the V2.5 grouped encoder architecture SVG."""
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
  <linearGradient id="gFourier" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#e74c3c"/>
    <stop offset="100%" stop-color="#c0392b"/>
  </linearGradient>
  <linearGradient id="gEntity" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#f39c12"/>
    <stop offset="100%" stop-color="#d68910"/>
  </linearGradient>
  <linearGradient id="gFusion" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#2c3e50"/>
    <stop offset="100%" stop-color="#1a252f"/>
  </linearGradient>
  <filter id="shadow">
    <feDropShadow dx="2" dy="2" stdDeviation="3" flood-color="#000" flood-opacity="0.5"/>
  </filter>
</defs>""")

    # Background
    lines.append(f'<rect width="{SVG_W}" height="{SVG_H}" fill="#0d0d15" rx="10"/>')

    # Title
    lines.append('<text x="600" y="30" text-anchor="middle" fill="#888" '
                 'font-family="Segoe UI,system-ui,sans-serif" font-size="14">'
                 'KaaN V3.0 — Grouped Encoder + Entity Attention + Aim Supervision</text>')

    # Helper: draw a layer node
    def node(x, y, w, h, label, sublabel, grad_id, dim_text=None):
        r = []
        r.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
                 f'fill="url(#{grad_id})" filter="url(#shadow)" opacity="0.9"/>')
        r.append(f'<text x="{x + w/2}" y="{y + h/2 - 4}" text-anchor="middle" '
                 f'fill="#fff" font-family="Segoe UI,system-ui,sans-serif" font-size="12" '
                 f'font-weight="bold">{label}</text>')
        r.append(f'<text x="{x + w/2}" y="{y + h/2 + 10}" text-anchor="middle" '
                 f'fill="rgba(255,255,255,0.7)" font-family="Segoe UI,system-ui,sans-serif" '
                 f'font-size="10">{sublabel}</text>')
        if dim_text:
            r.append(f'<text x="{x + w/2}" y="{y - 5}" text-anchor="middle" '
                     f'fill="#999" font-family="monospace" font-size="9">{dim_text}</text>')
        return '\n'.join(r)

    # Helper: draw a connection line
    def conn(x1, y1, x2, y2, color='#555'):
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{color}" stroke-width="1.5" opacity="0.6"/>')

    def arrow_conn(x1, y1, x2, y2, color='#555'):
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{color}" stroke-width="1.5" opacity="0.6" '
                f'marker-end="url(#arrowhead)"/>')

    # Node dimensions
    nw = 120
    nh = 44

    # ============================================================
    # SECTION 1: Input Observation (left column)
    # ============================================================
    obs_x = 30
    obs_y = 60
    lines.append(node(obs_x, obs_y, 100, 40, 'Observation', f'{OBS_DIM} dims', 'gInput', 'input'))

    # ============================================================
    # SECTION 2: Group Encoders (5 parallel streams)
    # ============================================================
    group_x = 170  # left edge of group encoder column
    enc_x = 340    # left edge of encoder output column
    gap = 58       # vertical gap between groups

    # --- Self State ---
    g0_y = 50
    lines.append(conn(obs_x + 100, obs_y + 20, group_x, g0_y + 20, '#8e44ad'))
    lines.append(node(group_x, g0_y, 140, nh, 'Self State', 'pos+vel+ang+stats+alive+walls', 'gInput', '32 raw'))

    # Fourier sub-node
    fourier_x = group_x + 10
    fourier_y = g0_y + nh + 2
    lines.append(f'<rect x="{fourier_x}" y="{fourier_y}" width="120" height="18" rx="3" '
                 f'fill="url(#gFourier)" opacity="0.7"/>')
    lines.append(f'<text x="{fourier_x + 60}" y="{fourier_y + 13}" text-anchor="middle" '
                 f'fill="#fff" font-family="monospace" font-size="9">Fourier pos(3→27)</text>')

    lines.append(conn(group_x + 140, g0_y + nh//2, enc_x, g0_y + nh//2, '#8e44ad'))
    lines.append(node(enc_x, g0_y, nw, nh, 'Self Encoder', 'Linear(56→96)+ReLU', 'gActor', '96 dims'))

    # --- Inventory ---
    g1_y = g0_y + gap + 12
    lines.append(conn(obs_x + 100, obs_y + 20, group_x, g1_y + 20, '#8e44ad'))
    lines.append(node(group_x, g1_y, 140, nh, 'Inventory', 'wep_1h+owned+ammo', 'gInput', '27 dims'))
    lines.append(conn(group_x + 140, g1_y + nh//2, enc_x, g1_y + nh//2, '#8e44ad'))
    lines.append(node(enc_x, g1_y, nw, nh, 'Inv Encoder', 'Linear(27→64)+ReLU', 'gActor', '64 dims'))

    # --- Enemy ---
    g2_y = g1_y + gap
    lines.append(conn(obs_x + 100, obs_y + 20, group_x, g2_y + 20, '#8e44ad'))
    lines.append(node(group_x, g2_y, 140, nh, 'Enemy', 'vis+xyz+dist+ang+wep+aim', 'gInput', '10 dims'))
    lines.append(conn(group_x + 140, g2_y + nh//2, enc_x, g2_y + nh//2, '#8e44ad'))
    lines.append(node(enc_x, g2_y, nw, nh, 'Enemy Encoder', 'Linear(10→64)+ReLU', 'gActor', '64 dims'))

    # --- Entities (with attention) ---
    g3_y = g2_y + gap
    lines.append(conn(obs_x + 100, obs_y + 20, group_x, g3_y + 20, '#8e44ad'))
    lines.append(node(group_x, g3_y, 140, nh, 'Entities (8x5)', '8 × [xyz+active+type]', 'gEntity', '40 dims'))
    lines.append(conn(group_x + 140, g3_y + nh//2, enc_x, g3_y + nh//2, '#f39c12'))
    lines.append(node(enc_x, g3_y, nw, nh, 'Entity Attn', 'shared(5→64)+pool', 'gEntity', '64 dims'))

    # Attention sub-label
    attn_y = g3_y + nh + 2
    lines.append(f'<rect x="{enc_x + 5}" y="{attn_y}" width="110" height="18" rx="3" '
                 f'fill="url(#gEntity)" opacity="0.6"/>')
    lines.append(f'<text x="{enc_x + 60}" y="{attn_y + 13}" text-anchor="middle" '
                 f'fill="#fff" font-family="monospace" font-size="9">softmax attention</text>')

    # --- Game State (passthrough) ---
    g4_y = g3_y + gap + 12
    lines.append(conn(obs_x + 100, obs_y + 20, group_x, g4_y + 20, '#8e44ad'))
    lines.append(node(group_x, g4_y, 140, nh, 'Game State', 'frags+enemy+time+dmg', 'gInput', '4 dims'))
    # passthrough arrow
    lines.append(conn(group_x + 140, g4_y + nh//2, enc_x + nw, g4_y + nh//2, '#666'))
    lines.append(f'<text x="{enc_x + 60}" y="{g4_y + nh//2 - 6}" text-anchor="middle" '
                 f'fill="#666" font-family="monospace" font-size="9">passthrough (4)</text>')

    # ============================================================
    # SECTION 3: Fusion (shared)
    # ============================================================
    fusion_x = enc_x + nw + 40
    fusion_y = (g0_y + g4_y + nh) // 2 - nh//2

    # Connection lines from all encoders to fusion
    for gy in [g0_y, g1_y, g2_y, g3_y, g4_y]:
        lines.append(conn(enc_x + nw, gy + nh//2, fusion_x, fusion_y + nh//2, '#555'))

    lines.append(node(fusion_x, fusion_y, 130, nh, 'Fusion Layer 1', 'Linear(292→256)+ReLU', 'gFusion', '292→256'))
    fusion2_y = fusion_y + nh + 12
    lines.append(conn(fusion_x + 65, fusion_y + nh, fusion_x + 65, fusion2_y, '#555'))
    lines.append(node(fusion_x, fusion2_y, 130, nh, 'Fusion Layer 2', 'Linear(256→256)+ReLU', 'gFusion', '256'))

    # ============================================================
    # SECTION 4: Split into Actor and Critic LSTMs
    # ============================================================
    split_x = fusion_x + 130 + 30
    lstm_w = 120
    lstm_h = nh

    # Actor LSTM
    actor_lstm_y = fusion2_y - 50
    lines.append(conn(fusion_x + 130, fusion2_y + nh//2, split_x, actor_lstm_y + lstm_h//2, '#3a7bd5'))
    lines.append(node(split_x, actor_lstm_y, lstm_w, lstm_h, 'Actor LSTM', '256→256', 'gActor', 'recurrent'))

    # Critic LSTM
    critic_lstm_y = fusion2_y + 50
    lines.append(conn(fusion_x + 130, fusion2_y + nh//2, split_x, critic_lstm_y + lstm_h//2, '#27ae60'))
    lines.append(node(split_x, critic_lstm_y, lstm_w, lstm_h, 'Critic LSTM', '256→256', 'gCritic', 'recurrent'))

    # ============================================================
    # SECTION 5: Heads and Outputs
    # ============================================================
    head_x = split_x + lstm_w + 30
    out_x = head_x + nw + 20

    # Actor head
    lines.append(conn(split_x + lstm_w, actor_lstm_y + lstm_h//2, head_x, actor_lstm_y + lstm_h//2, '#3a7bd5'))
    lines.append(node(head_x, actor_lstm_y, nw, lstm_h, 'Actor Head', 'Linear(256→18)', 'gActor', '18 dims'))

    # Output groups
    out_cont_y = actor_lstm_y - 50
    out_bin_y = actor_lstm_y
    out_weap_y = actor_lstm_y + 50

    lines.append(conn(head_x + nw, actor_lstm_y + lstm_h//4, out_x, out_cont_y + 20, '#e67e22'))
    lines.append(conn(head_x + nw, actor_lstm_y + lstm_h//2, out_x, out_bin_y + 20, '#e67e22'))
    lines.append(conn(head_x + nw, actor_lstm_y + 3*lstm_h//4, out_x, out_weap_y + 20, '#e67e22'))

    small_w = 160
    small_h = 36
    lines.append(node(out_x, out_cont_y, small_w, small_h, 'Continuous (4)', 'yaw,pitch,fwd,side', 'gHead'))
    lines.append(node(out_x, out_bin_y, small_w, small_h, 'Binary (3)', 'jump,crouch,attack', 'gHead'))
    lines.append(node(out_x, out_weap_y, small_w, small_h, 'Weapon (11)', 'Categorical', 'gHead'))

    # Log-std annotation
    lines.append(f'<text x="{out_x + small_w + 8}" y="{out_cont_y + 18}" fill="#e67e22" '
                 f'font-family="monospace" font-size="9">+ log_std</text>')

    # Critic head
    lines.append(conn(split_x + lstm_w, critic_lstm_y + lstm_h//2, head_x, critic_lstm_y + lstm_h//2, '#27ae60'))
    lines.append(node(head_x, critic_lstm_y, nw, lstm_h, 'Value Head', 'Linear(256→1)', 'gCritic', '1 dim'))

    lines.append(conn(head_x + nw, critic_lstm_y + lstm_h//2, out_x, critic_lstm_y + lstm_h//2, '#27ae60'))
    lines.append(node(out_x, critic_lstm_y, small_w, small_h, 'State Value', 'V(s)', 'gCritic'))

    # ============================================================
    # SECTION 6: Labels and annotations
    # ============================================================
    # "Shared" bracket for encoder + fusion
    bracket_x = enc_x - 5
    bracket_top = g0_y - 15
    bracket_bot = g4_y + nh + 15
    lines.append(f'<rect x="{bracket_x - 3}" y="{bracket_top}" width="{fusion_x + 130 - bracket_x + 8}" '
                 f'height="{bracket_bot - bracket_top}" rx="8" fill="none" '
                 f'stroke="#444" stroke-width="1" stroke-dasharray="4,3"/>')
    lines.append(f'<text x="{(bracket_x + fusion_x + 130) / 2}" y="{bracket_bot + 14}" '
                 f'text-anchor="middle" fill="#666" font-family="Segoe UI,system-ui,sans-serif" '
                 f'font-size="11" font-style="italic">Shared Encoder (actor + critic gradients)</text>')

    # "Separate" label
    sep_x = split_x + lstm_w // 2
    sep_y = max(out_weap_y + small_h + 20, critic_lstm_y + lstm_h + 20)
    lines.append(f'<text x="{sep_x + 50}" y="{sep_y}" '
                 f'text-anchor="middle" fill="#666" font-family="Segoe UI,system-ui,sans-serif" '
                 f'font-size="11" font-style="italic">Separate heads</text>')

    # ============================================================
    # SECTION 7: Parameter counts
    # ============================================================
    info_y = 520

    # Compute param counts per component
    fourier_params = 0  # computed, not learned
    self_enc_params = sum(p.numel() for p in model.self_encoder.parameters())
    inv_enc_params = sum(p.numel() for p in model.inv_encoder.parameters())
    enemy_enc_params = sum(p.numel() for p in model.enemy_encoder.parameters())
    entity_attn_params = sum(p.numel() for p in model.entity_attn.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    actor_lstm_params = sum(p.numel() for p in model.actor_lstm.parameters())
    critic_lstm_params = sum(p.numel() for p in model.critic_lstm.parameters())
    actor_head_params = sum(p.numel() for p in model.actor.parameters())
    critic_head_params = sum(p.numel() for p in model.critic.parameters())
    log_std_params = model.action_log_std.numel()

    param_info = [
        ('Fourier Enc', fourier_params, '#e74c3c'),
        ('Self Enc (56→96)', self_enc_params, '#3a7bd5'),
        ('Inv Enc (27→64)', inv_enc_params, '#3a7bd5'),
        ('Enemy Enc (10→64)', enemy_enc_params, '#3a7bd5'),
        ('Entity Attn (5→64)', entity_attn_params, '#f39c12'),
        ('Fusion (292→256→256)', fusion_params, '#888'),
        ('Actor LSTM', actor_lstm_params, '#3a7bd5'),
        ('Critic LSTM', critic_lstm_params, '#27ae60'),
        ('Actor Head', actor_head_params, '#3a7bd5'),
        ('Critic Head', critic_head_params, '#27ae60'),
        ('Log Std', log_std_params, '#e67e22'),
        ('Total', total_params, '#fff'),
    ]

    lines.append(f'<text x="40" y="{info_y}" fill="#888" font-family="Segoe UI,system-ui,sans-serif" '
                 f'font-size="13" font-weight="bold">Parameter Counts</text>')

    cols = 3
    for i, (plabel, pcount, pcolor) in enumerate(param_info):
        px = 40 + (i % cols) * 380
        py = info_y + 20 + (i // cols) * 20
        lines.append(f'<text x="{px}" y="{py}" fill="{pcolor}" '
                     f'font-family="monospace" font-size="11">'
                     f'{plabel}: {pcount:,}</text>')

    # ============================================================
    # SECTION 8: Data flow summary
    # ============================================================
    flow_y = info_y + 20 + (len(param_info) // cols + 1) * 20 + 15

    lines.append(f'<text x="40" y="{flow_y}" fill="#888" font-family="Segoe UI,system-ui,sans-serif" '
                 f'font-size="13" font-weight="bold">Data Flow</text>')

    flow_dims = [
        ('113', 'obs'),
        ('slice', '5 groups'),
        ('96+64+64+64+4', 'encoded'),
        ('292→256', 'fusion'),
        ('256→256', 'LSTM'),
        ('4+3+11=18', 'actions'),
    ]

    for i, (dim, lbl) in enumerate(flow_dims):
        fx = 40 + i * 180
        fy = flow_y + 15
        if i > 0:
            lines.append(f'<text x="{fx - 18}" y="{fy + 14}" fill="#555" '
                         f'font-family="monospace" font-size="14">&rarr;</text>')
        bw = len(dim) * 8 + 20
        lines.append(f'<rect x="{fx}" y="{fy}" width="{bw}" height="22" rx="4" '
                     f'fill="#1a1a2a" stroke="#444" stroke-width="1"/>')
        lines.append(f'<text x="{fx + bw/2}" y="{fy + 15}" text-anchor="middle" fill="#ddd" '
                     f'font-family="monospace" font-size="11">{dim}</text>')
        lines.append(f'<text x="{fx + bw/2}" y="{fy + 34}" text-anchor="middle" fill="#666" '
                     f'font-family="monospace" font-size="9">{lbl}</text>')

    # Init note
    lines.append(f'<text x="40" y="{SVG_H - 15}" fill="#555" '
                 f'font-family="Segoe UI,system-ui,sans-serif" font-size="10">'
                 f'Init: orthogonal (std=sqrt(2) encoders, std=0.01 actor head, std=1.0 critic head) '
                 f'| Fourier: fixed sin/cos L=4 bands | Entity attention: learned query vector</text>')

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
<title>KaaN V3.0 Neural Network Visualizer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #0a0a0f;
  color: #e0e0e0;
  padding: 20px;
  max-width: 1300px;
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

<h1>KaaN V3.0 -- Neural Network Visualizer</h1>
<p class="subtitle">Model: {html_mod.escape(ckpt_name)} | Generated: {now} | OBS_DIM: {OBS_DIM} | Grouped Encoder + Entity Attention + Fourier Position</p>

<div class="stat-bar">
  <div class="stat-item"><div class="val">{total_params:,}</div><div class="lbl">Total Parameters</div></div>
  <div class="stat-item"><div class="val">{OBS_DIM}</div><div class="lbl">Input Features</div></div>
  <div class="stat-item"><div class="val">18</div><div class="lbl">Action Outputs</div></div>
  <div class="stat-item"><div class="val">5</div><div class="lbl">Obs Groups</div></div>
  <div class="stat-item"><div class="val">2</div><div class="lbl">LSTMs (A/C)</div></div>
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
Yaw/pitch clamped to [-4.0, -2.5] for precise aim. Movement clamped to [-2.0, 0.5].
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
print(f"Log std values: {log_std}")
print(f"Binary biases: jump={actor_bias[4]:.2f} crouch={actor_bias[5]:.2f} attack={actor_bias[6]:.2f}")
print(f"Binary probs:  jump={binary_probs[0]:.4f} crouch={binary_probs[1]:.6f} attack={binary_probs[2]:.4f}")
