"""
AI Basketball Scout — Unified Pipeline
=======================================
Single entry point to run the full defensive analysis pipeline.

Usage:
    python run_pipeline.py                          # Run all steps
    python run_pipeline.py --video path/to/video    # Use custom video
    python run_pipeline.py --skip-detection          # Skip YOLO (reuse existing detections)
    python run_pipeline.py --report-only             # Just regenerate the report from existing data

Pipeline Steps:
    1. detect_players.py    — YOLO player detection + initial color extraction
    2. reextract_colors.py  — Improved jersey color extraction (torso band)
    3. classify_teams.py    — K-Means team classification with IQR filtering
    4. analyze_defense.py   — Defensive scheme analysis (possessions + GMM)
    5. validate_defense.py  — Generate annotated validation frames
    6. generate_report()    — Produce readable HTML report
"""

import subprocess
import sys
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = str(BASE_DIR / '.venv' / 'Scripts' / 'python.exe')
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
REPORT_DIR = PROCESSED_DIR / 'report'


def run_step(script_name, step_num, total_steps, description):
    """Run a pipeline step and capture output."""
    print(f"\n{'='*60}")
    print(f"  Step {step_num}/{total_steps}: {description}")
    print(f"  Script: {script_name}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [VENV_PYTHON, str(BASE_DIR / script_name)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start

    # Print stdout
    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")

    if result.returncode != 0:
        print(f"\n  *** FAILED (exit code {result.returncode}) ***")
        if result.stderr:
            print(f"  Error: {result.stderr[:500]}")
        return False, elapsed, result.stdout

    print(f"\n  [OK] Completed in {elapsed:.1f}s")
    return True, elapsed, result.stdout


def generate_report():
    """Generate an HTML report from the analysis results."""
    print(f"\n{'='*60}")
    print(f"  Generating Report")
    print(f"{'='*60}\n")

    # Load data
    analysis_path = PROCESSED_DIR / 'defensive_analysis.csv'
    if not analysis_path.exists():
        print("  Error: defensive_analysis.csv not found. Run analysis first.")
        return False

    df = pd.read_csv(analysis_path)

    # Possession-level stats — compute per-possession avg nopp dist from frame data
    poss_nopp = df.groupby('possession_id')['nearest_opp_mean'].mean().rename('poss_avg_nopp_dist')
    poss = df.groupby('possession_id').first()
    poss = poss.join(poss_nopp)
    live_schemes = ['man', 'zone', 'variant']
    live_poss = poss[poss['possession_scheme'].isin(live_schemes)]
    live_df = df[df['scheme'].isin(live_schemes)]

    # Scheme counts
    scheme_counts = poss['possession_scheme'].value_counts()
    man_count = scheme_counts.get('man', 0)
    zone_count = scheme_counts.get('zone', 0)
    variant_count = scheme_counts.get('variant', 0)

    # Determine scheme name
    if man_count > zone_count:
        scheme_name = "Man-to-Man"
        scheme_short = "man"
    elif zone_count > man_count:
        scheme_name = "Zone"
        scheme_short = "zone"
    else:
        scheme_name = "Mixed"
        scheme_short = "mixed"

    n_live = man_count + zone_count
    scheme_pct = 100 * n_live / max(len(live_poss), 1)

    # Compute key metrics
    avg_nopp_dist = live_df['nearest_opp_mean'].mean() if len(live_df) > 0 else 0
    avg_nov = live_df['nearest_opp_variance'].mean() if len(live_df) > 0 else 0
    avg_spacing = live_df['spacing_ratio'].mean() if len(live_df) > 0 else 0
    avg_spread = live_df['spread_ratio'].mean() if len(live_df) > 0 else 0

    # Find validation images
    val_dir = PROCESSED_DIR / 'validation'
    val_images = []
    if val_dir.exists():
        for i in range(1, 7):
            matches = list(val_dir.glob(f'sample_{i}_*.png'))
            if matches:
                val_images.append(matches[0])

    # Build HTML — clear old report first
    import shutil
    if REPORT_DIR.exists():
        shutil.rmtree(REPORT_DIR)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%B %d, %Y')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Defensive Scouting Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            background: #111;
            color: #d4d4d4;
            line-height: 1.7;
            font-size: 15px;
        }}
        .wrap {{ max-width: 880px; margin: 0 auto; padding: 32px 24px; }}

        /* Header */
        .header {{ padding: 48px 0 36px; border-bottom: 2px solid #222; margin-bottom: 40px; }}
        .header h1 {{ font-size: 24px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }}
        .header .date {{ color: #666; font-size: 13px; margin-top: 6px; }}

        /* Scheme callout */
        .scheme-callout {{
            background: linear-gradient(135deg, #1a2332 0%, #162029 100%);
            border: 1px solid #1e3a5f;
            border-radius: 10px;
            padding: 32px;
            margin-bottom: 36px;
        }}
        .scheme-callout .label {{ color: #5b8db8; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
        .scheme-callout .scheme {{ color: #fff; font-size: 36px; font-weight: 700; letter-spacing: -1px; }}
        .scheme-callout .detail {{ color: #7a9bb5; font-size: 14px; margin-top: 10px; line-height: 1.6; }}

        /* Stats row */
        .stats {{ display: flex; gap: 2px; margin-bottom: 36px; }}
        .stat {{
            flex: 1;
            background: #181818;
            padding: 20px;
            text-align: center;
        }}
        .stat:first-child {{ border-radius: 10px 0 0 10px; }}
        .stat:last-child {{ border-radius: 0 10px 10px 0; }}
        .stat .num {{ font-size: 28px; font-weight: 700; color: #fff; }}
        .stat .lbl {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}

        /* Sections */
        .section {{ margin-bottom: 36px; }}
        .section h2 {{ font-size: 16px; font-weight: 600; color: #999; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; }}

        /* Table */
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ padding: 8px 12px; text-align: left; color: #555; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #222; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #1a1a1a; font-size: 14px; }}
        tr:hover td {{ background: #161616; }}

        .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
        .tag-man {{ background: #1a2e44; color: #5b9bd5; }}
        .tag-zone {{ background: #3a1a1a; color: #d55b5b; }}
        .tag-variant {{ background: #332a1a; color: #d5a05b; }}

        /* Images */
        .frame-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
        .frame-grid img {{ width: 100%; border-radius: 6px; display: block; }}
        .plot-wrap {{ margin-top: 12px; }}
        .plot-wrap img {{ width: 100%; border-radius: 6px; }}

        /* Footer */
        .footer {{ padding: 32px 0; color: #444; font-size: 12px; text-align: center; border-top: 1px solid #1a1a1a; margin-top: 40px; }}

        /* Insight */
        .insight {{
            background: #161616;
            border-left: 3px solid #333;
            padding: 14px 18px;
            margin: 16px 0;
            font-size: 14px;
            color: #aaa;
        }}
    </style>
</head>
<body>
    <div class="wrap">
        <div class="header">
            <h1>Defensive Scouting Report</h1>
            <div class="date">{timestamp}</div>
        </div>

        <div class="scheme-callout">
            <div class="label">Identified Defensive Scheme</div>
            <div class="scheme">{scheme_name}</div>
            <div class="detail">
                Detected across {len(live_poss)} analyzed possessions.
                {"Defenders consistently maintain tight proximity to their assigned opponents, characteristic of man-to-man coverage." if scheme_short == "man" else "Defenders position themselves in areas of the court rather than tracking individual opponents."}
                {f'{variant_count} possession{"s" if variant_count != 1 else ""} showed unusual spacing patterns.' if variant_count > 0 else ''}
            </div>
        </div>

        <div class="stats">
            <div class="stat">
                <div class="num">{len(live_poss)}</div>
                <div class="lbl">Possessions Analyzed</div>
            </div>
            <div class="stat">
                <div class="num">{avg_nopp_dist:.0f}<span style="font-size:16px;color:#666">px</span></div>
                <div class="lbl">Avg Defender Distance</div>
            </div>
            <div class="stat">
                <div class="num">{avg_spread:.2f}</div>
                <div class="lbl">Coverage Ratio</div>
            </div>
        </div>"""

    # Possession detail table — only live play
    if len(live_poss) > 0:
        html += """
        <div class="section">
            <h2>Possession Breakdown</h2>
            <table>
                <tr><th>#</th><th>Classification</th><th>Frames</th><th>Defender Dist</th><th>Confidence</th></tr>"""

        for pid, row in live_poss.sort_values('possession_confidence', ascending=False).iterrows():
            scheme = row['possession_scheme']
            tag_class = f"tag-{scheme}"
            label = {'man': 'Man-to-Man', 'zone': 'Zone', 'variant': 'Variant'}.get(scheme, scheme.title())
            html += f"""
                <tr>
                    <td style="color:#555">{int(pid)}</td>
                    <td><span class="tag {tag_class}">{label}</span></td>
                    <td>{int(row['possession_n_frames'])}</td>
                    <td>{row['poss_avg_nopp_dist']:.0f}px</td>
                    <td>{row['possession_confidence']:.0%}</td>
                </tr>"""

        html += """
            </table>
        </div>"""

    # Metrics detail
    html += f"""
        <div class="section">
            <h2>Defensive Metrics</h2>
            <div class="insight">Metrics below are computed from live-play possessions only.</div>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
                <tr>
                    <td>Nearest-Opponent Distance</td>
                    <td><strong>{avg_nopp_dist:.1f} px</strong></td>
                    <td>{"Tight — defenders staying close to assignments" if avg_nopp_dist < 80 else "Moderate defensive spacing"}</td>
                </tr>
                <tr>
                    <td>Distance Consistency (NOV)</td>
                    <td><strong>{avg_nov:.0f}</strong></td>
                    <td>{"Consistent defender tracking" if avg_nov < 500 else "Some variation in defensive positioning"}</td>
                </tr>
                <tr>
                    <td>Defensive Coverage Ratio</td>
                    <td><strong>{avg_spread:.3f}</strong></td>
                    <td>{"Compact coverage — defense stays inside offensive spread" if avg_spread < 0.6 else "Wide coverage — defense mirrors offensive spread"}</td>
                </tr>
            </table>
        </div>"""

    # Analysis timeline
    plot_path = PROCESSED_DIR / 'defensive_analysis.png'
    if plot_path.exists():
        shutil.copy2(plot_path, REPORT_DIR / 'defensive_analysis.png')
        html += """
        <div class="section">
            <h2>Game Timeline</h2>
            <div class="plot-wrap">
                <img src="defensive_analysis.png" alt="Scheme analysis over time">
            </div>
        </div>"""

    # Validation frames
    if val_images:
        html += """
        <div class="section">
            <h2>Sample Frames</h2>
            <div class="frame-grid">"""
        for img in val_images[:6]:
            shutil.copy2(img, REPORT_DIR / img.name)
            html += f'\n                <img src="{img.name}" alt="{img.stem}">'
        html += """
            </div>
        </div>"""

    html += f"""
        <div class="footer">
            AI Basketball Scout &middot; {timestamp}
        </div>
    </div>
</body>
</html>"""

    report_path = REPORT_DIR / 'report.html'
    report_path.write_text(html, encoding='utf-8')
    print(f"  [OK] Report saved to: {report_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='AI Basketball Scout — Full Pipeline')
    parser.add_argument('--video', type=str, help='Path to game video (default: data/videos/game_trimmed.mp4)')
    parser.add_argument('--skip-detection', action='store_true',
                        help='Skip YOLO detection (reuse existing player_positions.csv)')
    parser.add_argument('--report-only', action='store_true',
                        help='Only regenerate the HTML report from existing data')
    args = parser.parse_args()

    print("=" * 60)
    print("  AI Basketball Scout - Defensive Analysis Pipeline")
    print("=" * 60)

    if args.report_only:
        generate_report()
        return

    # Determine total steps
    steps = []
    if not args.skip_detection:
        steps.append(('detect_players.py', 'YOLO Player Detection'))
        steps.append(('reextract_colors.py', 'Jersey Color Re-extraction'))
    steps.append(('classify_teams.py', 'Team Classification'))
    steps.append(('analyze_defense.py', 'Defensive Scheme Analysis'))
    steps.append(('validate_defense.py', 'Validation Frame Generation'))

    total = len(steps) + 1  # +1 for report
    timings = {}

    for i, (script, desc) in enumerate(steps, 1):
        ok, elapsed, output = run_step(script, i, total, desc)
        timings[script] = elapsed
        if not ok:
            print(f"\n*** Pipeline failed at step {i}: {script} ***")
            print("Fix the error and re-run. Use --skip-detection to skip YOLO if detections are already saved.")
            sys.exit(1)

    # Generate report
    generate_report()

    # Final summary
    total_time = sum(timings.values())
    print(f"\n{'='*60}")
    print(f"  [OK] Pipeline complete! Total time: {total_time:.1f}s")
    print(f"{'='*60}")
    print(f"\n  Timing breakdown:")
    for script, t in timings.items():
        print(f"    {script:<25} {t:.1f}s")
    print(f"\n  Report: {REPORT_DIR / 'report.html'}")
    print(f"  Data:   {PROCESSED_DIR / 'defensive_analysis.csv'}")
    print(f"  Plot:   {PROCESSED_DIR / 'defensive_analysis.png'}")


if __name__ == '__main__':
    main()
