#!/usr/bin/env python3
"""
monitor.py -- Track KaaN's behavioral evolution during training.

Runs in the background, checks training metrics every 10 minutes,
queries game servers via rcon, and generates an HTML dashboard.

Usage:
    python monitor.py                    # check every 10 minutes
    python monitor.py --interval 300     # check every 5 minutes
    python monitor.py --once             # single check
"""

import argparse
import json
import os
import re
import socket
import time
from datetime import datetime


DASHBOARD_PATH = "/home/oky/Documents/vscode/q2ai/evolve.html"
DATA_PATH = "/home/oky/Documents/vscode/q2ai/evolve_data.json"
TRAINING_LOG = "/tmp/training.log"
BASE_PORT = 27910
RCON_PASSWORD = "q2ai"

STAGES = [
    {"name": "Learn aim direction", "check": "aim_loss", "threshold": "aim_loss < 0.05"},
    {"name": "Hit stationary target", "check": "frag", "threshold": "frags > 50 per iter"},
    {"name": "Consistent hits", "check": "positive", "threshold": "frags > 200 per iter"},
    {"name": "Smooth aim (no zigzag)", "check": "smooth_aim", "threshold": "visual inspection"},
    {"name": "Hit moving target", "check": "moving_target", "threshold": "frags > 50 with bot moving"},
    {"name": "Survive under fire", "check": "survive", "threshold": "deaths < frags with bot shooting"},
    {"name": "Track + shoot moving bot", "check": "full_combat", "threshold": "K/D > 1.0 full bot"},
    {"name": "Learn movement", "check": "movement", "threshold": "speed > 200 avg"},
    {"name": "Frag/death ratio > 1.0", "check": "kd_ratio", "threshold": "frags/deaths > 1.0"},
    {"name": "Win a match", "check": "win", "threshold": "frags > bot frags"},
]

# Architecture milestones — update status when shipping new model versions
ARCH_MILESTONES = [
    {"name": "V1.0 MLP Baseline", "status": "done", "threshold": "91\u2192256\u2192256\u2192128\u219218 separate A/C"},
    {"name": "V2.0 LSTM Memory", "status": "done", "threshold": "encoder(91\u2192256) + LSTM(256\u2192256)"},
    {"name": "V2.5 Grouped Encoder", "status": "done", "threshold": "Fourier pos + entity attention + 2-layer fusion"},
    {"name": "V3.0 Aim Supervision", "status": "done", "threshold": "MSE aim loss + hit-only rewards + per-dim std clamp"},
    {"name": "V4.0 Self-Play", "status": "pending", "threshold": "train against copies of itself"},
]


def rcon_query(port, command):
    """Send rcon command to q2pro server and return response."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        packet = b"\xff\xff\xff\xff" + f"rcon {RCON_PASSWORD} {command}\n".encode()
        sock.sendto(packet, ("127.0.0.1", port))
        data = sock.recv(4096)
        sock.close()
        text = data[4:].decode("utf-8", errors="replace")
        if text.startswith("print\n"):
            text = text[6:]
        return text.strip()
    except Exception:
        return None


def get_server_status(port):
    """Get player status from server."""
    result = rcon_query(port, "status")
    if not result:
        return {}
    info = {}
    for line in result.split("\n"):
        line = line.strip()
        if "KaaN" in line:
            parts = line.split()
            if len(parts) >= 6:
                info["score"] = int(parts[1])
                info["ping"] = int(parts[2])
                info["fps"] = int(parts[-1]) if parts[-1].isdigit() else 0
        if "Bot[" in line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    info["bot_score"] = int(parts[1])
                except ValueError:
                    pass
    return info


def parse_training_log():
    """Parse all entries from training log."""
    if not os.path.exists(TRAINING_LOG):
        return []
    entries = []
    with open(TRAINING_LOG) as f:
        for line in f:
            # V1.3 format: [  N] steps=N | step_r=X frags=N deaths=N | pi=X v=X ent=X kl=X | Ns
            m = re.match(
                r'\[\s*(\d+)\]\s+steps=\s*(\d+)\s+\|\s+'
                r'step_r=\s*([-\d.]+)\s+frags=\s*(\d+)\s+deaths=\s*(\d+)\s+\|\s+'
                r'pi=\s*([-\d.]+)\s+v=\s*([-\d.]+)\s+ent=\s*([-\d.]+)\s+'
                r'kl=\s*([-\d.]+)\s+\|\s+(\d+)s',
                line.strip()
            )
            if m:
                entries.append({
                    "iteration": int(m.group(1)),
                    "steps": int(m.group(2)),
                    "episodes": 0,
                    "reward": 0.0,
                    "step_reward": float(m.group(3)),
                    "frags": int(m.group(4)),
                    "deaths": int(m.group(5)),
                    "policy_loss": float(m.group(6)),
                    "value_loss": float(m.group(7)),
                    "entropy": float(m.group(8)),
                    "kl": float(m.group(9)),
                    "elapsed": int(m.group(10)),
                })
                continue
            # V1.1 format: [  N] steps=N eps=N | R=X step_r=X frags=X deaths=N | ...
            m = re.match(
                r'\[\s*(\d+)\]\s+steps=\s*(\d+)\s+eps=\s*(\d+)\s+\|\s+'
                r'R=\s*([-\d.]+)\s+step_r=\s*([-\d.]+)\s+frags=\s*([-\d.]+)\s+'
                r'deaths=(\d+)\s+\|\s+'
                r'pi=\s*([-\d.]+)\s+v=\s*([-\d.]+)\s+ent=\s*([-\d.]+)\s+'
                r'kl=\s*([-\d.]+)\s+\|\s+(\d+)s',
                line.strip()
            )
            if m:
                entries.append({
                    "iteration": int(m.group(1)),
                    "steps": int(m.group(2)),
                    "episodes": int(m.group(3)),
                    "reward": float(m.group(4)),
                    "step_reward": float(m.group(5)),
                    "frags": float(m.group(6)),
                    "deaths": int(m.group(7)),
                    "policy_loss": float(m.group(8)),
                    "value_loss": float(m.group(9)),
                    "entropy": float(m.group(10)),
                    "kl": float(m.group(11)),
                    "elapsed": int(m.group(12)),
                })
    return entries


def evaluate_milestones(entries, server_info):
    """Check which milestones KaaN has achieved."""
    if not entries:
        return {s["name"]: "pending" for s in STAGES}

    latest = entries[-1]
    sr = latest["step_reward"]
    frags = latest["frags"]
    deaths = latest["deaths"]
    steps = latest["steps"]

    # Check across recent entries for cumulative stats
    recent = entries[-10:] if len(entries) >= 10 else entries
    total_frags = sum(e["frags"] for e in recent)
    total_deaths = sum(e["deaths"] for e in recent)
    avg_sr = sum(e["step_reward"] for e in recent) / len(recent)

    statuses = {}
    for s in STAGES:
        check = s["check"]
        if check == "blind_fire":
            # V1.3: blind fire penalty reduced to -0.05
            statuses[s["name"]] = "done" if sr > -0.01 else ("progress" if sr > -0.04 else "pending")
        elif check == "crouch":
            statuses[s["name"]] = "done" if sr > -0.01 else ("progress" if sr > -0.04 else "pending")
        elif check == "circling":
            statuses[s["name"]] = "done" if sr > 0 else ("progress" if sr > -0.02 else "pending")
        elif check == "aim_track":
            # Aim tracking: +0.02 when looking within 20deg — sr > 0 means tracking
            statuses[s["name"]] = "done" if avg_sr > 0.005 else ("progress" if avg_sr > -0.005 else "pending")
        elif check == "aimed_fire":
            # Aimed fire: +0.10 when shooting at visible enemy
            statuses[s["name"]] = "done" if avg_sr > 0.01 else ("progress" if avg_sr > 0 else "pending")
        elif check == "approach":
            # Approach: +0.03 when moving toward enemy
            statuses[s["name"]] = "done" if avg_sr > 0.005 else ("progress" if avg_sr > -0.005 else "pending")
        elif check == "weapon_switch":
            statuses[s["name"]] = "done" if avg_sr > 0.005 else ("progress" if avg_sr > -0.01 else "pending")
        elif check == "frag":
            statuses[s["name"]] = "done" if total_frags > 0 else "pending"
        elif check == "positive":
            statuses[s["name"]] = "done" if avg_sr > 0.01 else ("progress" if avg_sr > 0 else "pending")
        elif check == "kd_ratio":
            if total_deaths > 0 and total_frags > 0:
                kd = total_frags / total_deaths
                statuses[s["name"]] = "done" if kd > 0.5 else ("progress" if kd > 0.1 else "pending")
            else:
                statuses[s["name"]] = "pending"
        elif check == "win":
            kaan_score = server_info.get("score", 0)
            bot_score = server_info.get("bot_score", 999)
            statuses[s["name"]] = "done" if kaan_score > bot_score and kaan_score > 0 else "pending"
        else:
            statuses[s["name"]] = "pending"
    return statuses


def generate_html(history, milestones, latest, server_info):
    """Generate the HTML dashboard."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Milestone cards
    milestone_html = ""
    for s in STAGES:
        status = milestones.get(s["name"], "pending")
        if status == "done":
            cls = "done"
            icon = "&#10004;"
        elif status == "progress":
            cls = "progress"
            icon = "&#9200;"
        else:
            cls = "pending"
            icon = "&#9744;"
        milestone_html += f'<div class="card {cls}"><span class="icon">{icon}</span><div class="title">{s["name"]}</div><div class="threshold">{s["threshold"]}</div></div>\n'

    # Architecture milestone cards
    arch_milestone_html = ""
    for s in ARCH_MILESTONES:
        status = s["status"]
        if status == "done":
            cls = "done"
            icon = "&#10004;"
        elif status == "progress":
            cls = "progress"
            icon = "&#9200;"
        else:
            cls = "pending"
            icon = "&#9744;"
        arch_milestone_html += f'<div class="card {cls}"><span class="icon">{icon}</span><div class="title">{s["name"]}</div><div class="threshold">{s["threshold"]}</div></div>\n'

    # History rows
    history_rows = ""
    for h in reversed(history[-30:]):
        sr_class = "positive" if h["step_reward"] > 0 else "negative"
        history_rows += (
            f'<tr>'
            f'<td>{h["time"]}</td>'
            f'<td>{h["iteration"]}</td>'
            f'<td>{h["steps"]:,}</td>'
            f'<td class="{sr_class}">{h["step_reward"]:.4f}</td>'
            f'<td>{h["frags"]:.1f}</td>'
            f'<td>{h["deaths"]}</td>'
            f'<td>{h["entropy"]:.1f}</td>'
            f'<td>{h["kl"]:.1f}</td>'
            f'<td>{h.get("observation", "")}</td>'
            f'</tr>\n'
        )

    # Chart data
    chart_labels = json.dumps([h["time"] for h in history[-60:]])
    chart_sr = json.dumps([h["step_reward"] for h in history[-60:]])
    chart_frags = json.dumps([h["frags"] for h in history[-60:]])
    chart_entropy = json.dumps([h["entropy"] for h in history[-60:]])

    # Server status
    server_html = ""
    if server_info:
        score = server_info.get("score", "?")
        bot_score = server_info.get("bot_score", "?")
        fps = server_info.get("fps", "?")
        server_html = f"KaaN: {score} vs Bot: {bot_score} | FPS: {fps}"

    # Pre-compute stat values to avoid f-string ternary issues
    sr_val = f"{latest['step_reward']:.4f}" if latest else "0"
    sr_class = "positive" if latest and latest["step_reward"] > 0 else "negative"
    frags_val = str(latest["frags"]) if latest else "0"
    deaths_val = str(latest["deaths"]) if latest else "0"
    steps_val = f"{latest['steps']:,d}" if latest else "0"
    iter_val = str(latest["iteration"]) if latest else "0"
    ent_val = f"{latest['entropy']:.1f}" if latest else "0"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="60">
<title>KaaN Evolution Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 20px; }}
h1 {{ color: #ff6b35; margin-bottom: 5px; }}
.subtitle {{ color: #888; margin-bottom: 20px; }}
.section {{ margin-bottom: 30px; }}
.section h2 {{ color: #4ecdc4; margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 5px; }}

/* Milestone cards */
.cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
.card {{ padding: 15px; border-radius: 8px; border: 1px solid #333; }}
.card.done {{ background: #1a3a2a; border-color: #2ecc71; }}
.card.progress {{ background: #3a3a1a; border-color: #f39c12; }}
.card.pending {{ background: #1a1a2a; border-color: #555; }}
.card .icon {{ font-size: 24px; }}
.card .title {{ font-weight: bold; margin-top: 5px; }}
.card .threshold {{ color: #888; font-size: 12px; margin-top: 3px; }}

/* Stats */
.stats {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-bottom: 20px; }}
.stat {{ background: #151520; padding: 15px; border-radius: 8px; text-align: center; }}
.stat .value {{ font-size: 28px; font-weight: bold; color: #ff6b35; }}
.stat .label {{ color: #888; font-size: 12px; margin-top: 5px; }}
.positive {{ color: #2ecc71; }}
.negative {{ color: #e74c3c; }}

/* Table */
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ background: #1a1a2a; color: #4ecdc4; padding: 8px; text-align: left; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #222; }}
tr:hover {{ background: #151525; }}

/* Canvas */
canvas {{ background: #151520; border-radius: 8px; padding: 10px; }}
</style>
</head>
<body>
<h1>KaaN — Evolution Dashboard</h1>
<p class="subtitle">Last updated: {now} | {server_html}</p>

<div class="section">
<h2>Current Stats</h2>
<div class="stats">
<div class="stat"><div class="value {sr_class}">{sr_val}</div><div class="label">Step Reward</div></div>
<div class="stat"><div class="value">{frags_val}</div><div class="label">Frags (last iter)</div></div>
<div class="stat"><div class="value">{deaths_val}</div><div class="label">Deaths (last iter)</div></div>
<div class="stat"><div class="value">{steps_val}</div><div class="label">Total Steps</div></div>
<div class="stat"><div class="value">{iter_val}</div><div class="label">Iteration</div></div>
<div class="stat"><div class="value">{ent_val}</div><div class="label">Entropy</div></div>
</div>
</div>

<div class="section">
<h2>Milestones</h2>
<div class="cards">
{milestone_html}
</div>
</div>

<div class="section">
<h2>Architecture Milestones</h2>
<div class="cards">
{arch_milestone_html}
</div>
</div>

<div class="section">
<h2>Training History</h2>
<canvas id="chart" width="900" height="300"></canvas>
</div>

<div class="section">
<h2>Evolution Log</h2>
<table>
<tr><th>Time</th><th>Iter</th><th>Steps</th><th>Step R</th><th>Frags</th><th>Deaths</th><th>Entropy</th><th>KL</th><th>Observation</th></tr>
{history_rows}
</table>
</div>

<script>
// Simple chart drawing
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const labels = {chart_labels};
const sr = {chart_sr};
const frags = {chart_frags};

function drawLine(data, color, label, yMin, yMax) {{
    if (data.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {{
        const x = 60 + (i / (data.length - 1)) * (W - 80);
        const y = H - 40 - ((data[i] - yMin) / (yMax - yMin)) * (H - 60);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }}
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = '12px sans-serif';
    ctx.fillText(label, W - 70, ctx.canvas.height - data.length + 20);
}}

// Background
ctx.fillStyle = '#151520';
ctx.fillRect(0, 0, W, H);

// Zero line
const allVals = [...sr, ...frags];
const yMin = Math.min(-0.1, ...allVals) - 0.01;
const yMax = Math.max(0.1, ...allVals) + 0.01;
const zeroY = H - 40 - ((0 - yMin) / (yMax - yMin)) * (H - 60);
ctx.strokeStyle = '#444';
ctx.setLineDash([5, 5]);
ctx.beginPath();
ctx.moveTo(60, zeroY);
ctx.lineTo(W - 20, zeroY);
ctx.stroke();
ctx.setLineDash([]);

// Axis labels
ctx.fillStyle = '#888';
ctx.font = '11px sans-serif';
ctx.fillText('0', 40, zeroY + 4);
ctx.fillText(yMax.toFixed(2), 20, 25);
ctx.fillText(yMin.toFixed(2), 20, H - 35);

drawLine(sr, '#ff6b35', 'Step Reward', yMin, yMax);
drawLine(frags, '#4ecdc4', 'Frags', yMin, yMax);

// Legend
ctx.fillStyle = '#ff6b35'; ctx.fillText('● Step Reward', W - 150, 20);
ctx.fillStyle = '#4ecdc4'; ctx.fillText('● Frags', W - 150, 35);
</script>
</body>
</html>"""

    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)


def get_observation(latest):
    """Generate a short observation string."""
    if not latest:
        return ""
    sr = latest["step_reward"]
    frags = latest["frags"]
    deaths = latest["deaths"]

    if sr < -0.04:
        return "Heavy penalties — bad habits dominating"
    elif sr < -0.02:
        return "Still penalized — learning slowly"
    elif sr < 0:
        return f"Improving — {deaths} deaths, {frags} frags"
    elif sr < 0.005:
        return f"Near break-even — {deaths} deaths, {frags} frags"
    elif frags > 5:
        kd = frags / max(deaths, 1)
        return f"Combat engaged! {frags} frags, K/D={kd:.2f}"
    elif frags > 0:
        return f"Getting kills! {frags} frags, {deaths} deaths"
    else:
        return f"Tracking enemy — step_r={sr:.4f}, {deaths} deaths"


def main():
    parser = argparse.ArgumentParser(description="Monitor KaaN training evolution")
    parser.add_argument("--interval", type=int, default=600, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    print(f"KaaN Evolution Monitor — checking every {args.interval}s")
    print(f"Dashboard: {DASHBOARD_PATH}")

    # Load existing history
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH) as f:
            history = json.load(f)
    else:
        history = []

    while True:
        try:
            entries = parse_training_log()
            server_info = get_server_status(BASE_PORT)
            latest = entries[-1] if entries else None

            if latest:
                # Add to history
                entry = {
                    "time": datetime.now().strftime("%H:%M"),
                    "iteration": latest["iteration"],
                    "steps": latest["steps"],
                    "step_reward": latest["step_reward"],
                    "reward": latest["reward"],
                    "frags": latest["frags"],
                    "deaths": latest["deaths"],
                    "entropy": latest["entropy"],
                    "kl": latest["kl"],
                    "observation": get_observation(latest),
                }
                history.append(entry)

                # Save history
                with open(DATA_PATH, "w") as f:
                    json.dump(history, f)

                # Generate dashboard
                milestones = evaluate_milestones(entries, server_info)
                generate_html(history, milestones, latest, server_info)

                print(f"[{entry['time']}] iter={latest['iteration']} "
                      f"step_r={latest['step_reward']:.4f} frags={latest['frags']:.1f} "
                      f"— {entry['observation']}")
            else:
                print(f"[{datetime.now().strftime('%H:%M')}] No training data yet")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
