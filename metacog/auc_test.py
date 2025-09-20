import json
import numpy as np
from scipy import stats
import pandas as pd

def compare_external_influence(json1_paths, json2_paths, variant1_name='delegate', variant2_name='pass'):
    """
    Compare o_prob → y AUCs between two game variants across multiple file pairs.
    Lower AUC for o_prob means external factors are less predictive.
    """
    
    # Collect all observations across file pairs
    all_observations = []
    
    for json1_path, json2_path in zip(json1_paths, json2_paths):
        # Load JSONs
        with open(json1_path, 'r') as f:
            data1 = json.load(f)
        with open(json2_path, 'r') as f:
            data2 = json.load(f)
        
        # Extract o_prob AUCs for each model
        aucs1 = {}
        aucs2 = {}
        
        for model_name, model_data in data1.items():
            try:
                auc = model_data['capent']['univariate_choice_predictors']['o_prob']['mean_auc']
                aucs1[model_name] = auc
            except (KeyError, TypeError):
                print(f"Warning: Could not find o_prob AUC for {model_name} in {variant1_name} ({json1_path})")
        
        for model_name, model_data in data2.items():
            try:
                auc = model_data['capent']['univariate_choice_predictors']['o_prob']['mean_auc']
                aucs2[model_name] = auc
            except (KeyError, TypeError):
                print(f"Warning: Could not find o_prob AUC for {model_name} in {variant2_name} ({json2_path})")
        
        # Get paired data (only models present in both)
        common_models = sorted(set(aucs1.keys()) & set(aucs2.keys()))
        
        for model in common_models:
            all_observations.append({
                'model': model,
                'file_pair': f"{json1_path}-{json2_path}",
                f'{variant1_name}_auc': aucs1[model],
                f'{variant2_name}_auc': aucs2[model],
                'difference': aucs2[model] - aucs1[model]
            })
    
    if len(all_observations) == 0:
        print("Error: No common models found across any file pairs")
        return None
    
    # Convert to dataframe
    df = pd.DataFrame(all_observations)
    
    paired_aucs1 = df[f'{variant1_name}_auc'].values
    paired_aucs2 = df[f'{variant2_name}_auc'].values
    differences = df['difference'].values
    
    # Statistical tests
    # 1. Paired t-test
    t_stat, p_value_t = stats.ttest_rel(paired_aucs1, paired_aucs2)
    
    # 2. Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, p_value_w = stats.wilcoxon(paired_aucs1, paired_aucs2)
    
    # 3. Bootstrap confidence interval for mean difference
    n_bootstrap = 10000
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(differences), len(differences), replace=True)
        bootstrap_diffs.append(np.mean([differences[i] for i in sample_idx]))
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    # Summary statistics
    mean1 = np.mean(paired_aucs1)
    mean2 = np.mean(paired_aucs2)
    mean_diff = np.mean(differences)
    
    print(f"\n{'='*60}")
    print(f"EXTERNAL INFLUENCE COMPARISON (o_prob → choice)")
    print(f"{'='*60}")
    print(f"Number of observations: {len(all_observations)} (across {len(json1_paths)} file pairs)")
    print(f"\nMean AUC for {variant1_name}: {mean1:.4f}")
    print(f"Mean AUC for {variant2_name}: {mean2:.4f}")
    print(f"Mean difference ({variant2_name} - {variant1_name}): {mean_diff:.4f}")
    print(f"\nStatistical Tests:")
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_value_t:.4f}")
    print(f"  Wilcoxon test: W={w_stat:.1f}, p={p_value_w:.4f}")
    print(f"  95% CI for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    if mean_diff < 0:
        print(f"\n→ {variant2_name} shows LESS external influence (lower o_prob AUC)")
    elif mean_diff > 0:
        print(f"\n→ {variant1_name} shows LESS external influence (lower o_prob AUC)")
    else:
        print(f"\n→ No difference in external influence")
    
    print("\nPer-observation results:")
    print(df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print(f"INTROSPECTION COMPARISON (entropy → choice)")
    print(f"{'='*60}")
    
    entropy_observations = []
    
    for json1_path, json2_path in zip(json1_paths, json2_paths):
        with open(json1_path, 'r') as f:
            data1 = json.load(f)
        with open(json2_path, 'r') as f:
            data2 = json.load(f)
        
        for model_name in data1.keys():
            if model_name in data2:
                try:
                    auc1 = data1[model_name]['capent']['univariate_choice_predictors']['capabilities_entropy']['mean_auc']
                    auc2 = data2[model_name]['capent']['univariate_choice_predictors']['capabilities_entropy']['mean_auc']
                    entropy_observations.append({
                        'model': model_name,
                        f'{variant1_name}_entropy_auc': auc1,
                        f'{variant2_name}_entropy_auc': auc2,
                        'entropy_difference': auc2 - auc1
                    })
                except (KeyError, TypeError):
                    pass
    
    if entropy_observations:
        entropy_df = pd.DataFrame(entropy_observations)
        paired_entropy1 = entropy_df[f'{variant1_name}_entropy_auc'].values
        paired_entropy2 = entropy_df[f'{variant2_name}_entropy_auc'].values
        entropy_differences = entropy_df['entropy_difference'].values
        
        # Same three statistical tests as for o_prob
        # 1. Paired t-test
        t_stat_e, p_value_e_t = stats.ttest_rel(paired_entropy1, paired_entropy2)
        
        # 2. Wilcoxon signed-rank test
        w_stat_e, p_value_e_w = stats.wilcoxon(paired_entropy1, paired_entropy2)
        
        # 3. Bootstrap confidence interval
        bootstrap_diffs_e = []
        for _ in range(n_bootstrap):
            sample_idx = np.random.choice(len(entropy_differences), len(entropy_differences), replace=True)
            bootstrap_diffs_e.append(np.mean([entropy_differences[i] for i in sample_idx]))
        ci_lower_e = np.percentile(bootstrap_diffs_e, 2.5)
        ci_upper_e = np.percentile(bootstrap_diffs_e, 97.5)
        
        mean_e1 = np.mean(paired_entropy1)
        mean_e2 = np.mean(paired_entropy2)
        mean_diff_e = mean_e2 - mean_e1
        
        print(f"Number of observations: {len(entropy_observations)}")
        print(f"\nMean entropy AUC for {variant1_name}: {mean_e1:.4f}")
        print(f"Mean entropy AUC for {variant2_name}: {mean_e2:.4f}")
        print(f"Mean difference: {mean_diff_e:.4f}")
        print(f"\nStatistical Tests:")
        print(f"  Paired t-test: t={t_stat_e:.3f}, p={p_value_e_t:.4f}")
        print(f"  Wilcoxon test: W={w_stat_e:.1f}, p={p_value_e_w:.4f}")
        print(f"  95% CI for difference: [{ci_lower_e:.4f}, {ci_upper_e:.4f}]")
        
        if mean_diff_e > 0:
            print(f"\n→ {variant2_name} shows MORE introspection (higher entropy AUC)")
        elif mean_diff_e < 0:
            print(f"\n→ {variant1_name} shows MORE introspection (higher entropy AUC)")

        print("\nPer-observation results:")
        print(entropy_df.to_string(index=False))        

    return df, {
        'mean_diff': mean_diff,
        'p_value_t': p_value_t,
        'p_value_wilcoxon': p_value_w,
        'ci_95': (ci_lower, ci_upper)
    }

#results_df, stats_dict = compare_external_influence(['res_dicts_factual_sa_dg.json','res_dicts_reasoning_sa_dg.json'], ['res_dicts_factual_sa_aop.json','res_dicts_reasoning_sa_aop.json'])
results_df, stats_dict = compare_external_influence(['res_dicts_factual_mc_dg.json','res_dicts_reasoning_mc_dg.json'], ['res_dicts_factual_mc_aop.json','res_dicts_reasoning_mc_aop.json'])
#results_df, stats_dict = compare_external_influence(['res_dicts_factual_sa_dg.json','res_dicts_reasoning_sa_dg.json','res_dicts_factual_mc_dg.json','res_dicts_reasoning_mc_dg.json'], ['res_dicts_factual_sa_aop.json','res_dicts_reasoning_sa_aop.json','res_dicts_factual_mc_aop.json','res_dicts_reasoning_mc_aop.json'])
print("\nOverall Statistics:", stats_dict)
#print("\nDetailed Results DataFrame:")
#print(results_df)