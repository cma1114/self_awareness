import json
import matplotlib.pyplot as plt
import numpy as np

def make_plots(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    # --- Chart 1: capabilities_entropy_vs_delegate_choice_ctrl_surface+o_prob ---

    # Extract data for the first chart
    chart1_data = {}
    for model, values in data.items():
        if 'capent' in values and values['capent'] is not None:
            try:
                chart1_data[model] = {
                    'partial_r': values['capent']['adjusted_influence']['capabilities_entropy_vs_delegate_choice_ctrl_surface+o_prob']['partial_r'],
                    'ci_lower': values['capent']['adjusted_influence']['capabilities_entropy_vs_delegate_choice_ctrl_surface+o_prob']['ci_lo'],
                    'ci_upper': values['capent']['adjusted_influence']['capabilities_entropy_vs_delegate_choice_ctrl_surface+o_prob']['ci_hi']
                }
            except KeyError:
                # Skip models that don't have the required nested keys
                continue

    # Prepare data for plotting
    models = list(chart1_data.keys())
    partial_r_values = [d['partial_r'] for d in chart1_data.values()]
    ci_lower = [d['partial_r'] - d['ci_lower'] for d in chart1_data.values()]
    ci_upper = [d['ci_upper'] - d['partial_r'] for d in chart1_data.values()]
    error_bars = [ci_lower, ci_upper]

    # Create the first bar chart
    plt.figure(figsize=(12, 7))
    plt.bar(models, partial_r_values, yerr=error_bars, capsize=5, color='skyblue', edgecolor='black')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel("Partial Correlation (r)", fontsize=12)
    plt.title("Model Capability Entropy vs. Delegate Choice (Controlled)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    out_filename = filename.replace('.json', '_capent_dg_plots.png')
    plt.savefig(out_filename)

    # --- Chart 2: dy_minus_dx1 point values ---

    # Extract data for the second chart
    chart2_data = {}
    for model, values in data.items():
        if 'capent' in values and values['capent'] is not None:
            try:
                chart2_data[model] = {
                    'point': values['capent']['differences_adjusted']['dy_minus_dx1']['point'],
                    'ci_lower': values['capent']['differences_adjusted']['dy_minus_dx1']['lo'],
                    'ci_upper': values['capent']['differences_adjusted']['dy_minus_dx1']['hi']
                }
            except KeyError:
                # Skip models that don't have the required nested keys
                continue

    # Prepare data for plotting
    models_chart2 = list(chart2_data.keys())
    point_values = [d['point'] for d in chart2_data.values()]
    ci_lower_chart2 = [d['point'] - d['ci_lower'] for d in chart2_data.values()]
    ci_upper_chart2 = [d['ci_upper'] - d['point'] for d in chart2_data.values()]
    error_bars_chart2 = [ci_lower_chart2, ci_upper_chart2]

    # Create the second bar chart
    plt.figure(figsize=(12, 7))
    plt.bar(models_chart2, point_values, yerr=error_bars_chart2, capsize=5, color='lightgreen', edgecolor='black')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel("Point Value", fontsize=12)
    plt.title("Adjusted Differences (dy_minus_dx1)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    out_filename = filename.replace('.json', '_game_vs_stated_plots.png')
    plt.savefig(out_filename)

make_plots('res_dicts_factual_sa_dg.json')