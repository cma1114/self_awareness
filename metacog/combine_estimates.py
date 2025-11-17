import numpy as np

def pool_partial_corr_seeds(seed_tuples, alpha=0.05):
    """
    seed_tuples: list of (r, L, U) where L,U are 95% CIs for r
    Returns: {'r_pooled', 'ci', 'prediction_interval', 'I2'}
    """
    # Optional: use scipy if available for critical values
    try:
        from scipy import stats
        zcrit = stats.norm.ppf(1 - alpha/2)
        def tcrit(df): return stats.t.ppf(1 - alpha/2, df=max(df, 1))
    except Exception:
        zcrit = 1.959963984540054  # 95% normal
        def tcrit(df): return zcrit

    eps = 1e-12
    def clip_r(x): return np.clip(x, -1 + eps, 1 - eps)
    atanh = np.arctanh; tanh = np.tanh

    if len(seed_tuples) == 0:
        raise ValueError("Provide at least one (r, L, U).")

    # Per-seed Fisher z and SE from CI width
    r = np.array([clip_r(s[0]) for s in seed_tuples], float)
    L = np.array([clip_r(s[1]) for s in seed_tuples], float)
    U = np.array([clip_r(s[2]) for s in seed_tuples], float)
    z = atanh(r); Lz = atanh(L); Uz = atanh(U)
    se_z = (Uz - Lz) / (2 * zcrit)
    vi = np.maximum(se_z**2, eps)
    k = len(z)

    # k == 1: return that seed
    if k == 1:
        return {
            'r_pooled': float(r[0]),
            'ci': (float(L[0]), float(U[0])),
            'prediction_interval': (float(L[0]), float(U[0])),
            'I2': 0.0
        }

    # Fixed-effect weights for heterogeneity stats
    w0 = 1.0 / vi
    zbar0 = np.sum(w0 * z) / np.sum(w0)
    Q = np.sum(w0 * (z - zbar0)**2)
    C = np.sum(w0) - (np.sum(w0**2) / np.sum(w0))
    tau2 = max((Q - (k - 1)) / max(C, eps), 0.0)

    # Random-effects pooling
    w = 1.0 / (vi + tau2)
    zbar = np.sum(w * z) / np.sum(w)
    se_pooled = np.sqrt(1.0 / np.sum(w))

    # CI and prediction interval on z scale
    ci_z = (zbar - zcrit * se_pooled, zbar + zcrit * se_pooled)
    pred_se = np.sqrt(tau2 + se_pooled**2)
    pi_z = (zbar - tcrit(k - 2) * pred_se, zbar + tcrit(k - 2) * pred_se)

    # Back-transform and I^2
    r_pooled = tanh(zbar)
    ci = (tanh(ci_z[0]), tanh(ci_z[1]))
    pi = (tanh(pi_z[0]), tanh(pi_z[1]))
    I2 = float(max((Q - (k - 1)) / max(Q, eps), 0.0) * 100.0)

    return {
        'r_pooled': float(r_pooled),
        'ci': (float(ci[0]), float(ci[1])),
        'prediction_interval': (float(pi[0]), float(pi[1])),
        'I2': I2
    }

all_data=[]
datadict= {
    'dataset': "SimpleMC",
    'model': "Grok 3",
    'data': [(0.21, 0.12, 0.30), (0.28, 0.19, 0.36), (0.13, 0.03, 0.22)]
}
all_data.append(datadict)
datadict= {
    'dataset': "SimpleMC",
    'model': "Sonnet 4",
    'data': [(0.19, 0.1, 0.28), (0.16, 0.06, 0.25), (0.17, 0.08, 0.26)]
}
all_data.append(datadict)
datadict= {
    'dataset': "SimpleMC",
    'model': "GPT-4o Mini",
    'data': [(0.01, -0.08, 0.10), (-0.04, -0.13, 0.05), (0.01, -0.09, 0.10)]
}
all_data.append(datadict)
datadict= {
    'dataset': "SimpleMC",
    'model': "Gem 2 Flash",
    'data': [(-0.04, -0.13, 0.05), (-0.04, -0.13, 0.05), (0.00, -0.1, 0.09)]
}
all_data.append(datadict)
datadict= {
    'dataset': "GPSA",
    'model': "GPT-4o",
    'data': [(0.25, 0.15, 0.34), (0.22, 0.17, 0.31), (0.19, 0.09, 0.28)]
}
all_data.append(datadict)
datadict= {
    'dataset': "GPSA",
    'model': "GPT-5",
    'data': [(0.25, 0.15, 0.34), (0.24, 0.15, 0.34), (0.13, 0.03, 0.23)]
}
all_data.append(datadict)
datadict= {
    'dataset': "GPSA",
    'model': "DeepSeek Chat",
    'data': [(0.07, -0.03, 0.17), (0.15, 0.05, 0.25), (0.09, -0.01, 0.19)]
}
all_data.append(datadict)
datadict= {
    'dataset': "GPSA",
    'model': "Gem 2 Flash",
    'data': [(0.01, -0.09, 0.12), (0.08, -0.02, 0.18), (0.04, -0.06, 0.15)]
}
all_data.append(datadict)
for entry in all_data:
    result = pool_partial_corr_seeds(entry['data'])
#    print(f"Dataset: {entry['dataset']}, Model: {entry['model']}, Pooled r: {result['r_pooled']:.4g}, 95% CI: ({result['ci'][0]:.4g}, {result['ci'][1]:.4g})")
#    print(f"Dataset: {entry['dataset']}, Model: {entry['model']}, Pooled r: {result['r_pooled']:.4g}, 95% CI: ({result['ci'][0]:.4g}, {result['ci'][1]:.4g}), Prediction Interval: ({result['prediction_interval'][0]:.4g}, {result['prediction_interval'][1]:.4g}), I²: {result['I2']:.2f}%")
#    {"dataset": "SimpleMC", "model_name": "Grok 3", "point_estimate": 0.2084, "CI": (0.80, 0.90)},
    print(f'    {{"dataset": "{entry["dataset"]}", "model_name": "{entry["model"]}", "point_estimate": {result["r_pooled"]:.4g}, "CI": ({result["ci"][0]:.4g}, {result["ci"][1]:.4g})}},')