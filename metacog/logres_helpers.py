import statsmodels.stats.proportion as smp
import scipy.stats as ss
import numpy as np
import re

import math
from scipy.stats import binomtest, rankdata, pointbiserialr, norm, pearsonr, chi2
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

def compare_predictors_of_choice_simple(X1, X2, X3, y, continuous_controls=None, categorical_controls=None, normvars=True):
    ret_str = ""
    results_dict = {}
    try:
        #### Setup ####
        original_names = {
            'X1': X1.name or 'X1', 
            'X2': X2.name or 'X2', 
            'X3': X3.name or 'X3', 
            'y': y.name or 'y'
        }
        
        # Build base DataFrame
        df = pd.DataFrame({
            'X1': X1.values, 
            'X2': X2.values, 
            'X3': X3.values, 
            'y': y.values
        })
        
        # Add control variables if provided
        cont_control_names = []
        if continuous_controls:
            for ctrl in continuous_controls:
                ctrl_name = ctrl.name or f'cont_control_{len(cont_control_names)}'
                cont_control_names.append(ctrl_name)
                df[ctrl_name] = ctrl.values
        
        cat_control_names = []
        if categorical_controls:
            for ctrl in categorical_controls:
                ctrl_name = ctrl.name or f'cat_control_{len(cat_control_names)}'
                cat_control_names.append(ctrl_name)
                df[ctrl_name] = ctrl.values
        
        df = df.dropna(subset=['X1', 'X2', 'X3', 'y'])
        
        # Normalize X1, X2, X3 and any continuous controls
        scaler = StandardScaler()
        vars_to_standardize = ['X1', 'X2', 'X3'] + cont_control_names
        df_norm = df.copy()
        if normvars: df_norm[vars_to_standardize] = scaler.fit_transform(df[vars_to_standardize])
        ##################################
        
        #### Relationships among Stated Self/Other Confidences, Entropy, and Choice ####
        # CORRELATIONS
        ret_str += "Q1: Which test (Stated Self, Stated Other, Game) is most influenced by entropy\n"
        r_X1_X2, p_X1_X2 = pearsonr(df_norm['X1'], df_norm['X2'])
        r_X1_X3, p_X1_X3 = pearsonr(df_norm['X1'], df_norm['X3'])
        r_X2_X3, p_X2_X3 = pearsonr(df_norm['X2'], df_norm['X3'])
        r_X1_Y, p_X1_Y = pearsonr(df_norm['X1'], df_norm['y'])
        r_X2_Y, p_X2_Y = pearsonr(df_norm['X2'], df_norm['y'])
        r_X3_Y, p_X3_Y = pearsonr(df_norm['X3'], df_norm['y'])
        
        ret_str += f"Pearson Correlations:\n"
        ret_str += f"  {original_names['X1']}-{original_names['X2']}: r={r_X1_X2:.3f}, p={p_X1_X2:.4f}\n"
        ret_str += f"  {original_names['X1']}-{original_names['X3']}: r={r_X1_X3:.3f}, p={p_X1_X3:.4f}\n"
        ret_str += f"  {original_names['X2']}-{original_names['X3']}: r={r_X2_X3:.3f}, p={p_X2_X3:.4f}\n"
        ret_str += f"  {original_names['X1']}-{original_names['y']}: r={r_X1_Y:.3f}, p={p_X1_Y:.4f}\n"
        ret_str += f"  {original_names['X2']}-{original_names['y']}: r={r_X2_Y:.3f}, p={p_X2_Y:.4f}\n"
        ret_str += f"  {original_names['X3']}-{original_names['y']}: r={r_X3_Y:.3f}, p={p_X3_Y:.4f}\n"
        ret_str += "\n"

        results_dict['pearson_correlations'] = {
            f'{original_names["X1"]}-{original_names["X2"]}': {'r': float(r_X1_X2), 'p': float(p_X1_X2)},
            f'{original_names["X1"]}-{original_names["X3"]}': {'r': float(r_X1_X3), 'p': float(p_X1_X3)},
            f'{original_names["X2"]}-{original_names["X3"]}': {'r': float(r_X2_X3), 'p': float(p_X2_X3)},
            f'{original_names["X1"]}-{original_names["y"]}': {'r': float(r_X1_Y), 'p': float(p_X1_Y)},
            f'{original_names["X2"]}-{original_names["y"]}': {'r': float(r_X2_Y), 'p': float(p_X2_Y)},
            f'{original_names["X3"]}-{original_names["y"]}': {'r': float(r_X3_Y), 'p': float(p_X3_Y)},
        }

        # SPEARMAN CORRELATIONS for X3 relationships 
        rho_X3_X1, p_rho_X3_X1 = spearmanr(df['X3'], df['X1'])
        rho_X3_X2, p_rho_X3_X2 = spearmanr(df['X3'], df['X2'])
        rho_X3_Y, p_rho_X3_Y = spearmanr(df['X3'], df['y'])
        
        ret_str += f"Spearman ρ({original_names['X3']},{original_names['X1']}) = {rho_X3_X1:.3f} (p={p_rho_X3_X1:.4f})\n"
        ret_str += f"Spearman ρ({original_names['X3']},{original_names['X2']}) = {rho_X3_X2:.3f} (p={p_rho_X3_X2:.4f})\n"
        ret_str += f"Spearman ρ({original_names['X3']},{original_names['y']}) = {rho_X3_Y:.3f} (p={p_rho_X3_Y:.4f})\n"
        ret_str += "\n"

        results_dict['spearman_correlations'] = {
            f'{original_names["X3"]}-{original_names["X1"]}': {'rho': float(rho_X3_X1), 'p': float(p_rho_X3_X1)},
            f'{original_names["X3"]}-{original_names["X2"]}': {'rho': float(rho_X3_X2), 'p': float(p_rho_X3_X2)},
            f'{original_names["X3"]}-{original_names["y"]}': {'rho': float(rho_X3_Y), 'p': float(p_rho_X3_Y)},
        }

        # PATH ANALYSIS - asymmetry test
        model_X3_to_X1 = LinearRegression().fit(df_norm[['X3', 'X2']], df_norm['X1'])
        r2_X3_to_X1 = model_X3_to_X1.score(df_norm[['X3', 'X2']], df_norm['X1'])
        
        model_X1_to_X3 = LinearRegression().fit(df_norm[['X1', 'X2']], df_norm['X3'])
        r2_X1_to_X3 = model_X1_to_X3.score(df_norm[['X1', 'X2']], df_norm['X3'])
        
        ret_str += f"{original_names['X3']}+{original_names['X2']}→{original_names['X1']}: R²={r2_X3_to_X1:.3f}\n"
        ret_str += f"{original_names['X1']}+{original_names['X2']}→{original_names['X3']}: R²={r2_X1_to_X3:.3f}\n"
        ret_str += "\n"

        results_dict['asymmetry_test'] = {
            f"{original_names['X3']}_{original_names['X2']}_to_{original_names['X1']}_R2": float(r2_X3_to_X1),
            f"{original_names['X1']}_{original_names['X2']}_to_{original_names['X3']}_R2": float(r2_X1_to_X3),
        }

        ret_str += f"Comparative entropy impacts\n"
        import statsmodels.api as sm
        from scipy import stats
        dfm = df_norm.dropna(subset=['X1','X2','X3','y']).copy()
        # 1) X3 → X1 controlling for X2 (OLS with robust SEs)
        X_ols = sm.add_constant(dfm[['X2', 'X3']])
        ols = sm.OLS(dfm['X1'], X_ols).fit(cov_type='HC3')
        beta_X3_to_X1 = ols.params['X3']
        p_X3_to_X1 = ols.pvalues['X3']

        # 2) X3 → y controlling for X2 (Logit with unpenalized MLE)
        X_logit = sm.add_constant(dfm[['X2', 'X3']])
        logit = sm.Logit(dfm['y'], X_logit).fit(disp=0)
        beta_X3_to_y = logit.params['X3']
        p_X3_to_y = logit.pvalues['X3']

        # Robustness 1: Add y as control in OLS predicting X1
        X_ols_full = sm.add_constant(dfm[['X2','y','X3']])
        ols_full = sm.OLS(dfm['X1'], X_ols_full).fit(cov_type='HC3')
        beta_X3_to_X1_full = ols_full.params['X3']
        p_X3_to_X1_full = ols_full.pvalues['X3']

        # Robustness 2: y ~ X1 + X2 + X3
        X_logit_full = sm.add_constant(dfm[['X1','X2','X3']])
        logit_full = sm.Logit(dfm['y'], X_logit_full).fit(disp=0)
        beta_X3_to_y_full = logit_full.params['X3']
        p_X3_to_y_full = logit_full.pvalues['X3']

        # Store results
        results_dict['comparative_entropy_impacts'] = {
            'X3_to_X1_controlling_X2': {'beta': float(beta_X3_to_X1), 'p': float(p_X3_to_X1)},
            'X3_to_Y_controlling_X2': {'beta': float(beta_X3_to_y), 'p': float(p_X3_to_y)},
            'X3_to_X1_controlling_X2_Y': {'beta': float(beta_X3_to_X1_full), 'p': float(p_X3_to_X1_full)},
            'X3_to_Y_controlling_X1_X2': {'beta': float(beta_X3_to_y_full), 'p': float(p_X3_to_y_full)},
        }
        ret_str += f"Controlling for {original_names['X2']}:\n"
        ret_str += f"  {original_names['X3']}→{original_names['X1']}: β = {beta_X3_to_X1:.4f}, p = {p_X3_to_X1:.4f}\n"
        ret_str += f"  {original_names['X3']}→{original_names['y']}: β = {beta_X3_to_y:.4f}, p = {p_X3_to_y:.4f}\n"
        ret_str += "\n"
        
        ret_str += f"Controlling for BOTH {original_names['X2']} and {original_names['y']}:\n"
        ret_str += f"  {original_names['X3']}→{original_names['X1']}: β = {beta_X3_to_X1_full:.4f}, p = {p_X3_to_X1_full:.4f}\n"
        ret_str += f"Controlling for BOTH {original_names['X1']} and {original_names['X2']}:\n"
        ret_str += f"  {original_names['X3']}→{original_names['y']}: β = {beta_X3_to_y_full:.4f}, p = {p_X3_to_y_full:.4f}\n"
        ret_str += "\n"

        # Effect size comparison for Q1
        # Partial R^2 for X1: compare base (X2 only) vs full (X2 + X3)
        Xb_ols = sm.add_constant(dfm[['X2']])
        ols_b = sm.OLS(dfm['X1'], Xb_ols).fit()
        Xf_ols = sm.add_constant(dfm[['X2','X3']])
        ols_f = sm.OLS(dfm['X1'], Xf_ols).fit()
        partial_R2_X3_on_X1 = 1.0 - (ols_f.ssr / ols_b.ssr)

        # Tjur's R^2 for y (interpretable under imbalance)
        def tjur_R2(result, X, y_vec):
            p = result.predict(X)
            return float(p[y_vec==1].mean() - p[y_vec==0].mean())

        Xb_log = sm.add_constant(dfm[['X2']])
        mb = sm.Logit(dfm['y'], Xb_log).fit(disp=0)
        Xf_log = sm.add_constant(dfm[['X2','X3']])
        mf = sm.Logit(dfm['y'], Xf_log).fit(disp=0)
        delta_tjur_y = tjur_R2(mf, Xf_log, dfm['y']) - tjur_R2(mb, Xb_log, dfm['y'])

        results_dict.setdefault('comparative_entropy_impacts', {}).update({
            'partial_R2_X3_on_X1_ctrl_X2': float(partial_R2_X3_on_X1),
            'delta_TjurR2_X3_on_y_ctrl_X2': float(delta_tjur_y),
        })
        ret_str += f"Effect sizes (controls: {original_names['X2']}): partial R² (X1)={partial_R2_X3_on_X1:.4f}, ΔTjur R² (y)={delta_tjur_y:.4f}\n"
        ##################################

        #### Analysis of Stated Self/Other Confidences and Entropy as Predictors of Choice ####
        # Univariate analysis - fit separate logistic regression for each predictor
        ret_str += "Q2: Which factors drive game performance?\n"

        univar = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for var in ['X1','X2','X3']:
            Xi = sm.add_constant(dfm[[var]])
            fit = sm.Logit(dfm['y'], Xi).fit(disp=0)
            coef = float(fit.params[var])
            pval = float(fit.pvalues[var])
            # AUC with sklearn (unpenalized if available)
            clf = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
            auc = cross_val_score(clf, dfm[[var]].values, dfm['y'].values, cv=cv, scoring='roc_auc')
            univar[var] = {'coefficient': coef, 'odds_ratio': np.exp(coef), 'p_value': pval,
                        'mean_auc': float(auc.mean()), 'std_auc': float(auc.std()), 'aic': float(fit.aic)}

        results_dict['univariate_choice_predictors'] = { original_names[k]: v for k,v in univar.items() }

        ret_str += "="*60
        ret_str += "\nUNIVARIATE RESULTS (each predictor alone)\n"
        ret_str += "="*60 + "\n"
        results_df = pd.DataFrame(univar).T
        results_df = results_df.round(4)
        ret_str += results_df[['coefficient', 'odds_ratio', 'p_value', 'mean_auc', 'aic']].to_string()
        ret_str += "\n"

        # LIKELIHOOD RATIO TESTS - which variables add value?
        ret_str += "="*60 + "\n"
        ret_str += "LIKELIHOOD RATIO TESTS\n"
        ret_str += "="*60 + "\n"

        def lr_test(base_vars, add_var):
            Xb = sm.add_constant(dfm[base_vars])
            Xf = sm.add_constant(dfm[base_vars + [add_var]])
            mb = sm.Logit(dfm['y'], Xb).fit(disp=0)
            mf = sm.Logit(dfm['y'], Xf).fit(disp=0)
            lr = 2*(mf.llf - mb.llf)
            p = stats.chi2.sf(lr, df=1)
            return lr, p

        for base_vars, add_var in [(['X1'], 'X2'), (['X1'], 'X3'), (['X2'], 'X3'),
                                (['X1','X2'], 'X3'), (['X1','X3'], 'X2')]:
            lr, p = lr_test(base_vars, add_var)
            
            base_names = [original_names[v] for v in base_vars]
            ret_str += f"{original_names[add_var]} adds to {'+'.join(base_names)}: LR={lr:.3f}, p={p:.4f}\n"

            if 'likelihood_ratio_tests' not in results_dict:
                results_dict['likelihood_ratio_tests'] = {}
            
            key = f"{original_names[add_var]}_adds_to_{'+'.join(base_names)}"
            results_dict['likelihood_ratio_tests'][key] = {
                'LR': float(lr), 
                'p': float(p)
            } 

        # REGRESSION WITH CONTROLS 
        if cont_control_names or cat_control_names:
            ret_str += "\n"
            ret_str += "="*60 + "\n"
            ret_str += "FULL MODEL WITH CONTROLS\n"
            ret_str += "="*60 + "\n"
            
            # Build formula string
            formula_parts = [original_names['y'], '~', original_names['X1'], '+', original_names['X2'], '+', original_names['X3']]
            
            # Add continuous controls
            for ctrl in cont_control_names:
                formula_parts.extend(['+', ctrl])
            
            # Add categorical controls with C() notation
            for ctrl in cat_control_names:
                formula_parts.extend(['+', f'C({ctrl})'])
            
            formula = ' '.join(formula_parts)
            
            # Use original df with actual column names for statsmodels
            df_for_model = pd.DataFrame()
            df_for_model[original_names['X1']] = df_norm['X1']
            df_for_model[original_names['X2']] = df_norm['X2']
            df_for_model[original_names['X3']] = df_norm['X3']
            df_for_model[original_names['y']] = df_norm['y']
            
            # Add controls (standardized continuous, original categorical)
            for ctrl in cont_control_names:
                df_for_model[ctrl] = df_norm[ctrl]
            for ctrl in cat_control_names:
                df_for_model[ctrl] = df[ctrl]
            
            model = smf.logit(formula, data=df_for_model)
            result = model.fit(disp=0)
            ret_str += result.summary().as_text()

            results_dict['full_model_choice_predictors'] = {}
            for var_key in ['X1', 'X2', 'X3']:
                var_name = original_names[var_key]
                if var_name in result.params.index:
                    results_dict['full_model_choice_predictors'][var_name] = {
                        'coef': float(result.params[var_name]),
                        'p': float(result.pvalues[var_name]),
                    }
        ##################################
        
    except Exception as e:
        ret_str += f"Error in simplified entropy analysis: {str(e)}\n"
        
    return ret_str, results_dict

def compare_predictors_of_choice(X1, X2, X3, y):
    ret_str = ""
    try:
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'y': y})
        df = df[['X1', 'X2', 'X3', 'y']].dropna()
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(df[['X1', 'X2', 'X3']])

        df_norm = pd.DataFrame(X_normalized, columns=['X1', 'X2', 'X3'])
        df_norm['y'] = df['y'].values

        ret_str += "="*60
        r_X1_X2, p_X1_X2 = pearsonr(df_norm['X1'], df_norm['X2'])
        r_X1_X3, p_X1_X3 = pearsonr(df_norm['X1'], df_norm['X3'])
        r_X2_X3, p_X2_X3 = pearsonr(df_norm['X2'], df_norm['X3'])

        ret_str += f"Correlations:\n"
        ret_str += f"Pearson X1-X2: r={r_X1_X2:.3f}, p={p_X1_X2:.4f}\n"
        ret_str += f"Pearson X1-X3: r={r_X1_X3:.3f}, p={p_X1_X3:.4f}\n"
        ret_str += f"Pearson X2-X3: r={r_X2_X3:.3f}, p={p_X2_X3:.4f}\n"
        ret_str += "\n"

        # Step 3: Univariate analysis - fit separate logistic regression for each predictor
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for var in ['X1', 'X2', 'X3']:
            # Fit model
            X = df_norm[[var]].values
            y = df_norm['y'].values
            
            model = LogisticRegression(solver='liblinear')
            model.fit(X, y)
            
            # Get coefficient and p-value
            coef = model.coef_[0, 0]
            z_score = coef / (np.sqrt(np.diag(np.linalg.inv(X.T @ X))) * 0.5)  # Approximate SE
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
            
            # Calculate AUC with cross-validation
            auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            mean_auc = auc_scores.mean()
            std_auc = auc_scores.std()
            
            # Calculate log-likelihood and AIC
            probs = model.predict_proba(X)[:, 1]
            log_likelihood = np.sum(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
            aic = -2 * log_likelihood + 2 * 2  # 2 parameters (intercept + coefficient)
            
            results[var] = {
                'coefficient': coef,
                'odds_ratio': np.exp(coef),
                'p_value': p_value,
                'mean_auc': mean_auc,
                'std_auc': std_auc,
                'aic': aic,
                'log_likelihood': log_likelihood
            }

        # Step 4: Multivariate analysis - all predictors together
        X_all = df_norm[['X1', 'X2', 'X3']].values
        y = df_norm['y'].values

        model_full = LogisticRegression(solver='liblinear')
        model_full.fit(X_all, y)

        # Calculate VIF (Variance Inflation Factors) to check multicollinearity
        from numpy.linalg import inv
        corr_matrix = df_norm[['X1', 'X2', 'X3']].corr().values
        vif = np.diag(inv(corr_matrix))

        # Get multivariate results
        multi_results = {}
        for i, var in enumerate(['X1', 'X2', 'X3']):
            multi_results[var] = {
                'multi_coefficient': model_full.coef_[0, i],
                'multi_odds_ratio': np.exp(model_full.coef_[0, i]),
                'vif': vif[i]
            }

        ret_str += "="*60
        ret_str += "\nUNIVARIATE RESULTS (each predictor alone)\n"
        ret_str += "="*60 + "\n"
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(4)
        ret_str += results_df[['coefficient', 'odds_ratio', 'p_value', 'mean_auc', 'aic']].to_string()
        ret_str += "\n"

        ret_str += "="*60
        ret_str += "\nMULTIVARIATE RESULTS (all predictors together)\n"
        ret_str += "="*60 + "\n"
        multi_df = pd.DataFrame(multi_results).T
        multi_df = multi_df.round(4)
        ret_str += multi_df.to_string()
        ret_str += "\n"

        ret_str += "="*60
        ret_str += "\nBEST PREDICTOR DETERMINATION\n"
        ret_str += "="*60 + "\n"

        # Find best by AUC
        best_auc = max(results.keys(), key=lambda x: results[x]['mean_auc'])
        ret_str += f"Highest AUC (univariate): {best_auc} with AUC = {results[best_auc]['mean_auc']:.4f}\n"

        # Find best by AIC (lowest is better)
        best_aic = min(results.keys(), key=lambda x: results[x]['aic'])
        ret_str += f"Lowest AIC (univariate): {best_aic} with AIC = {results[best_aic]['aic']:.2f}\n"

        # Find best by absolute coefficient in multivariate model
        best_multi = max(['X1', 'X2', 'X3'], key=lambda x: abs(multi_results[x]['multi_coefficient']))
        ret_str += f"Largest coefficient (multivariate): {best_multi} with coef = {multi_results[best_multi]['multi_coefficient']:.4f}\n"

        ret_str += "\n"
        ret_str += "="*60
        ret_str += "\nFINAL ANSWER\n"
        ret_str += "="*60 + "\n"

        # Determine overall best predictor based on multiple criteria
        scores = {var: 0 for var in ['X1', 'X2', 'X3']}
        scores[best_auc] += 1
        scores[best_aic] += 1
        scores[best_multi] += 1

        best_overall = max(scores.keys(), key=lambda x: scores[x])

        if max(scores.values()) >= 2:
            ret_str += f"BEST PREDICTOR: {best_overall}\n"
            ret_str += f"Reason: Best in {scores[best_overall]} out of 3 criteria\n"
        else:
            ret_str += f"BEST PREDICTOR BY AUC: {best_auc}\n"
            ret_str += "Note: No clear winner across all criteria, but AUC is most reliable for prediction\n"

        # Step 7: Additional validation - compare nested models
        ret_str += "\n"
        ret_str += "="*60 + "\n"
        ret_str += "LIKELIHOOD RATIO TESTS (does adding other variables help?)\n"
        ret_str += "="*60 + "\n"

        # For the best predictor, test if adding others improves significantly
        X_best = df_norm[[best_auc]].values
        model_best = LogisticRegression(solver='liblinear').fit(X_best, y)
        ll_best = np.sum(y * np.log(model_best.predict_proba(X_best)[:, 1] + 1e-10) + 
                        (1 - y) * np.log(1 - model_best.predict_proba(X_best)[:, 1] + 1e-10))

        # Test adding each other variable
        for var in ['X1', 'X2', 'X3']:
            if var != best_auc:
                X_combined = df_norm[[best_auc, var]].values
                model_combined = LogisticRegression(solver='liblinear').fit(X_combined, y)
                ll_combined = np.sum(y * np.log(model_combined.predict_proba(X_combined)[:, 1] + 1e-10) + 
                                (1 - y) * np.log(1 - model_combined.predict_proba(X_combined)[:, 1] + 1e-10))
                
                lr_stat = 2 * (ll_combined - ll_best)
                p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
                
                ret_str += f"Adding {var} to {best_auc}: LR stat = {lr_stat:.3f}, p = {p_value:.4f}\n"
                if p_value < 0.05:
                    ret_str += f"  -> {var} adds significant predictive value\n"
                else:
                    ret_str += f"  -> {var} does NOT add significant predictive value\n"

        from itertools import combinations
        variables = ['X1', 'X2', 'X3']
        y = df['y'].values

        for base_size in [1, 2]:
            for base_vars in combinations(variables, base_size):
                base_vars = list(base_vars)
                X_base = df[base_vars].values
                model_base = LogisticRegression(solver='liblinear').fit(X_base, y)
                
                # Calculate log-likelihood for base model
                probs_base = model_base.predict_proba(X_base)[:, 1]
                probs_base = np.clip(probs_base, 1e-10, 1-1e-10)  # Avoid log(0)
                ll_base = np.sum(y * np.log(probs_base) + (1 - y) * np.log(1 - probs_base))
                
                for add_var in variables:
                    if add_var not in base_vars:
                        X_full = df[base_vars + [add_var]].values
                        model_full = LogisticRegression(solver='liblinear').fit(X_full, y)
                        
                        # Calculate log-likelihood for full model
                        probs_full = model_full.predict_proba(X_full)[:, 1]
                        probs_full = np.clip(probs_full, 1e-10, 1-1e-10)
                        ll_full = np.sum(y * np.log(probs_full) + (1 - y) * np.log(1 - probs_full))
                        
                        lr_stat = 2 * (ll_full - ll_base)
                        p_val = 1 - stats.chi2.cdf(lr_stat, df=1)
                        
                        ret_str += f"{add_var} adds to {'+'.join(base_vars)}: LR={lr_stat:.3f}, p={p_val:.4f}\n"



        # Compare R² (variance explained)
        # For X3→X1 (continuous)
        model_X3_X1 = LinearRegression().fit(df[['X3']], df['X1'])
        r2_X3_X1 = model_X3_X1.score(df[['X3']], df['X1'])

        # For X3→Y (pseudo-R² for binary)
        model_X3_Y = LogisticRegression().fit(df[['X3']], df['y'])
        # McFadden's pseudo-R²
        null_model = LogisticRegression().fit(np.ones((len(df), 1)), df['y'])
        ll_null = np.sum(df['y'] * np.log(null_model.predict_proba(np.ones((len(df), 1)))[:, 1] + 1e-10) + 
                        (1 - df['y']) * np.log(1 - null_model.predict_proba(np.ones((len(df), 1)))[:, 1] + 1e-10))
        ll_model = np.sum(df['y'] * np.log(model_X3_Y.predict_proba(df[['X3']])[:, 1] + 1e-10) + 
                        (1 - df['y']) * np.log(1 - model_X3_Y.predict_proba(df[['X3']])[:, 1] + 1e-10))
        pseudo_r2_X3_Y = 1 - (ll_model / ll_null)

        # Also compare Spearman correlations (rank-based, works for both)
        rho_X3_X1, p_rho_X3_X1 = spearmanr(df['X3'], df['X1'])
        rho_X3_Y, p_rho_X3_Y = spearmanr(df['X3'], df['y'])

        ret_str += f"X3 explains {r2_X3_X1:.1%} of variance in X1\n"
        ret_str += f"X3 explains {pseudo_r2_X3_Y:.1%} of variance in Y (pseudo-R²)\n"
        ret_str += f"Spearman ρ(X3,X1) = {rho_X3_X1:.3f} (p={p_rho_X3_X1:.4f})\n"
        ret_str += f"Spearman ρ(X3,Y) = {rho_X3_Y:.3f} (p={p_rho_X3_Y:.4f})\n"

        # X3 predicting X1, controlling for X2
        model_X3_to_X1 = LinearRegression().fit(df[['X3', 'X2']], df['X1'])
        r2_X3_to_X1 = model_X3_to_X1.score(df[['X3', 'X2']], df['X1'])

        # X1 predicting X3, controlling for X2  
        model_X1_to_X3 = LinearRegression().fit(df[['X1', 'X2']], df['X3'])
        r2_X1_to_X3 = model_X1_to_X3.score(df[['X1', 'X2']], df['X3'])

        ret_str += f"X3+X2→X1: R²={r2_X3_to_X1:.3f}\n"
        ret_str += f"X1+X2→X3: R²={r2_X1_to_X3:.3f}\n"


        # X3 predicting X1, controlling for X2
        model = LinearRegression().fit(df[['X2', 'X3']], df['X1'])
        X3_coef = model.coef_[1]

        # Force float64 to prevent object dtype issues
        X = df[['X2', 'X3']].values
        X = X.astype(np.float64)  # THIS IS THE FIX

        y_pred = model.predict(df[['X2', 'X3']])  # Use original df here
        residuals = df['X1'].values - y_pred
        n = len(df)
        p = X.shape[1]
        mse = np.sum(residuals**2) / (n - p)
        var_coef = mse * np.linalg.inv(X.T @ X)[1, 1]
        se = np.sqrt(var_coef)
        t_stat = X3_coef / se
        p_value_X1 = 2 * (1 - stats.t.cdf(abs(t_stat), n - p))

        ret_str += f"Controlling for X2:\n"
        ret_str += f"  X3→X1: β = {X3_coef:.4f}, p = {p_value_X1:.4f}\n"

        # X3 predicting Y, controlling for X2  
        model_y = LogisticRegression(solver='liblinear').fit(df[['X2', 'X3']], df['y'])
        X3_coef_y = model_y.coef_[0, 1]  # X3 is second predictor

        # For logistic regression p-value (approximate)
        z_stat = X3_coef_y / (np.sqrt(np.diag(np.linalg.inv(X.T @ X))[1]) * 0.5)
        p_value_Y = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        ret_str += f"  X3→Y: β = {X3_coef_y:.4f}, p = {p_value_Y:.4f}\n"

        # X3 predicting X1, controlling for BOTH X2 and Y
        model_X1_full = LinearRegression().fit(df[['X2', 'y', 'X3']], df['X1'])
        X3_coef_X1_full = model_X1_full.coef_[2]  # X3 is third predictor

        # Calculate standard error for significance test
        X_full = df[['X2', 'y', 'X3']].values.astype(np.float64)
        y_pred = model_X1_full.predict(df[['X2', 'y', 'X3']])
        residuals = df['X1'].values - y_pred
        n = len(df)
        p = X_full.shape[1]
        mse = np.sum(residuals**2) / (n - p)
        var_coef = mse * np.linalg.inv(X_full.T @ X_full)[2, 2]
        se = np.sqrt(var_coef)
        t_stat = X3_coef_X1_full / se
        p_value_X1_full = 2 * (1 - stats.t.cdf(abs(t_stat), n - p))

        ret_str += f"Controlling for BOTH X2 and Y:\n"
        ret_str += f"  X3→X1: β = {X3_coef_X1_full:.4f}, p = {p_value_X1_full:.4f}\n"

        # X3 predicting Y, controlling for BOTH X1 and X2
        model_full = LogisticRegression(solver='liblinear').fit(df[['X1', 'X2', 'X3']], df['y'])
        X3_coef_full = model_full.coef_[0, 2]  # X3 is third predictor

        # For significance test
        X_full = df[['X1', 'X2', 'X3']].values.astype(np.float64)
        # Approximate z-test for logistic coefficient
        z_stat = X3_coef_full / (np.sqrt(np.diag(np.linalg.inv(X_full.T @ X_full))[2]) * 0.5)
        p_value_full = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        ret_str += f"Controlling for BOTH X1 and X2:\n"
        ret_str += f"  X3→Y: β = {X3_coef_full:.4f}, p = {p_value_full:.4f}\n"


        # X3→X1 relationship
        model_simple = LinearRegression().fit(df[['X3']], df['X1'])
        r2_simple = model_simple.score(df[['X3']], df['X1'])

        model_controlled = LinearRegression().fit(df[['X3', 'X2']], df['X1'])
        r2_controlled = model_controlled.score(df[['X3', 'X2']], df['X1'])  

        model_X2_only = LinearRegression().fit(df[['X2']], df['X1'])
        r2_X2_only = model_X2_only.score(df[['X2']], df['X1'])

        partial_r2 = r2_controlled - r2_X2_only  

        attenuation_X1 = (r2_simple - partial_r2) / r2_simple

        # X3→Y relationship
        # Simple model: just X3
        model_Y_simple = LogisticRegression(solver='liblinear').fit(df[['X3']], df['y'])
        prob_simple = np.clip(model_Y_simple.predict_proba(df[['X3']])[:, 1], 1e-10, 1-1e-10)
        ll_simple = np.sum(df['y'] * np.log(prob_simple) + (1 - df['y']) * np.log(1 - prob_simple))

        # Null model (intercept only) for pseudo-R² calculation
        model_null = LogisticRegression(solver='liblinear').fit(np.ones((len(df), 1)), df['y'])
        prob_null = np.clip(model_null.predict_proba(np.ones((len(df), 1)))[:, 1], 1e-10, 1-1e-10)
        ll_null = np.sum(df['y'] * np.log(prob_null) + (1 - df['y']) * np.log(1 - prob_null))

        # Controlled model: X3 and X2
        model_Y_controlled = LogisticRegression(solver='liblinear').fit(df[['X3', 'X2']], df['y'])
        prob_controlled = np.clip(model_Y_controlled.predict_proba(df[['X3', 'X2']])[:, 1], 1e-10, 1-1e-10)
        ll_controlled = np.sum(df['y'] * np.log(prob_controlled) + (1 - df['y']) * np.log(1 - prob_controlled))

        # Model with just X2
        model_Y_X2_only = LogisticRegression(solver='liblinear').fit(df[['X2']], df['y'])
        prob_X2_only = np.clip(model_Y_X2_only.predict_proba(df[['X2']])[:, 1], 1e-10, 1-1e-10)
        ll_X2_only = np.sum(df['y'] * np.log(prob_X2_only) + (1 - df['y']) * np.log(1 - prob_X2_only))

        # Calculate partial contribution of X3 (beyond X2)
        # This is the improvement in log-likelihood from adding X3 to X2
        ll_improvement_from_X3 = ll_controlled - ll_X2_only
        ll_improvement_simple = ll_simple - ll_null

        # Attenuation: how much does X3's effect weaken when controlling for X2?
        attenuation_Y = 1 - (ll_improvement_from_X3 / ll_improvement_simple)

        ret_str += f"X3→X1 attenuates {attenuation_X1:.1%} when controlling for X2\n"
        ret_str += f"X3→Y attenuates {attenuation_Y:.1%} when controlling for X2\n"
        if attenuation_Y < attenuation_X1:
            ret_str += "X3→Y is MORE robust to controlling for difficulty (X2)\n"
            ret_str += "This suggests delegation uses internal signals beyond external difficulty cues\n"
        else:
            ret_str += "X3→X1 is MORE robust to controlling for difficulty (X2)\n"

    except Exception as e:
        ret_str += f"Error during analysis: {e}"
    return ret_str


def contingency(delegate: np.ndarray, correct: np.ndarray):
    """
    delegate : bool[N]   True -> model delegated
    correct  : bool[N]   True -> model would be correct on its own
    returns  : TP, FN, FP, TN as ints
    """
    TP = np.sum(delegate  & ~correct)   # delegate & wrong
    FN = np.sum(~delegate & ~correct)   # keep     & wrong
    FP = np.sum(delegate  &  correct)   # delegate & right
    TN = np.sum(~delegate &  correct)   # keep     & right
    return TP, FN, FP, TN

def lift_mcc_stats(tp, fn, fp, tn, kept_correct, p0, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)

    # ---------- point estimates --------------------------------------------
    k         = len(kept_correct)                    # ★ kept items
    kept_acc  = kept_correct.mean() if k else np.nan # ★ acc from real data
    lift      = kept_acc - p0

    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc   = (tp*tn - fp*fn) / denom if denom else np.nan

    # ---------- p-values ----------------------------------------------------
    p_lift = binomtest(kept_correct.sum(), k, p0,  # ★ successes = real kept
                       alternative='two-sided').pvalue
    p_mcc  = mcnemar([[tn, fp],
                      [fn, tp]], exact=True).pvalue

    # ---------- bootstrap CIs ----------------------------------------------
    N        = tp + fn + fp + tn
    counts   = np.array([tp, fn, fp, tn], int)
    probs    = counts / N

    lifts, mccs = [], []
    kept_idx = np.arange(k)                         # ★ indices for kept vector

    for _ in range(n_boot):
        # ----- lift: resample ONLY kept correctness ----------------★
        b_k_acc = kept_correct[rng.choice(kept_idx, k, replace=True)].mean()
        lifts.append(b_k_acc - p0)

        # ----- MCC: multinomial resample of 4-cell table (unchanged)
        sample = rng.choice(4, size=N, replace=True, p=probs)
        btp, bfn, bfp, btn = np.bincount(sample, minlength=4)
        bden = math.sqrt((btp+bfp)*(btp+bfn)*(btn+bfp)*(btn+bfn))
        bmcc = (btp*btn - bfp*bfn) / bden if bden else 0.0
        mccs.append(bmcc)

    ci_lift = tuple(np.percentile(lifts, [2.5, 97.5]))  # ★ tuple()
    ci_mcc  = tuple(np.percentile(mccs,  [2.5, 97.5]))

    boot_arr = np.array(mccs)
    p_mcc = 2 * min((boot_arr <= 0).mean(), (boot_arr >= 0).mean())
    
    return dict(lift=lift,   lift_ci=ci_lift,  p_lift=p_lift,
                mcc=mcc,     mcc_ci =ci_mcc,  p_mcc =p_mcc)

def self_acc_stats(cap_corr, team_corr, kept_mask):
    k           = kept_mask.sum()                
    s           = team_corr[kept_mask].sum()      
    p0          = cap_corr.mean()    

    p_val = ss.binomtest(s, k, p0, alternative='two-sided').pvalue
    lo, hi = smp.proportion_confint(s, k, alpha=0.05, method='wilson')
    lift    = s/k - p0
    lift_lo = lo   - p0
    lift_hi = hi   - p0
    return lift, lift_lo, lift_hi, p_val

def self_acc_stats_boot(baseline_correct, kept_correct, kept_mask, n_boot=2000, seed=0):
    """
    baseline_correct : 1/0[N]   baseline correctness for *every* item
    kept_correct     : 1/0[N]   correctness *actually achieved in game*
    kept_mask        : bool[N]  True where the model answered itself
    """
    A_base = np.nanmean(baseline_correct)
    A_kept = kept_correct[kept_mask].mean() if kept_mask.any() else np.nan
    lift   = A_kept - A_base

    # paired bootstrap
    rng  = np.random.default_rng(seed)
    idx0 = np.arange(len(baseline_correct))
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(idx0, len(idx0), replace=True)
        A_b = np.nanmean(baseline_correct[idx])
        km  = kept_mask[idx]
        A_k = kept_correct[idx][km].mean() if km.any() else 0
        boots.append(A_k - A_b)

    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two  = 2 * min(np.mean(np.array(boots) <= 0),
                     np.mean(np.array(boots) >= 0))
    return lift, lo, hi, p_two

def delegate_gap_stats(TP, FN, FP, TN):
    def wilson(p, n, alpha=0.05):
        return smp.proportion_confint(count=p*n, nobs=n, alpha=alpha, method='wilson')

    n_wrong, n_right = TP+FN, FP+TN
    p_del_wrong  = TP / n_wrong
    p_del_right  = FP / n_right
    delta_d      = p_del_wrong - p_del_right

    lo1, hi1 = wilson(p_del_wrong,  n_wrong)   # CI for P(delegate|wrong)
    lo2, hi2 = wilson(p_del_right,  n_right)   # CI for P(delegate|right)
    ci_low  = delta_d - np.sqrt((p_del_wrong-lo1)**2 + (hi2-p_del_right)**2)
    ci_high = delta_d + np.sqrt((hi1-p_del_wrong)**2 + (p_del_right-lo2)**2)

    table = [[TP, FN],   # rows: wrong/right ; cols: delegate/keep
            [FP, TN]]
    chi2, p_val, *_ = ss.chi2_contingency(table, correction=False)
    return delta_d, ci_low, ci_high, p_val

def mcc_ci_boot(TP, FN, FP, TN):
    mcc = (TP*TN - FP*FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    score = (mcc + 1)/2

    N = TP+FN+FP+TN
    wrong = np.array([1]*TP + [1]*FN + [0]*FP + [0]*TN, dtype=bool)   # 1 = model would be wrong
    dele  = np.array([1]*TP + [0]*FN + [1]*FP + [0]*TN, dtype=bool)   # 1 = model delegated

    boot=[]
    rng = np.random.default_rng(0)
    for _ in range(2000):
        idx = rng.choice(N, N, replace=True)
        tp = np.sum(wrong[idx] &  dele[idx])
        tn = np.sum(~wrong[idx] & ~dele[idx])
        fp = np.sum(~wrong[idx] &  dele[idx])
        fn = np.sum(wrong[idx] & ~dele[idx])
        denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        boot_mcc = (tp*tn - fp*fn)/denom if denom else 0
        boot.append(boot_mcc)
    ci = np.percentile(boot, [2.5,97.5])
    return mcc, score, ci

LOG_METRICS_TO_EXTRACT = [
    "Delegation to teammate occurred",
    "Phase 1 self-accuracy (from completed results, total - phase2)",
    "Phase 2 self-accuracy",
    "Statistical test (P2 self vs P1)"
]

LOG_METRIC_PATTERNS = {
    "Delegation to teammate occurred": re.compile(r"^\s*Delegation to teammate occurred in (.*)$"),
    "Phase 1 self-accuracy (from completed results, total - phase2)": re.compile(r"^\s*Phase 1 self-accuracy \(from completed results, total - phase2\): (.*)$"),
    "Phase 2 self-accuracy": re.compile(r"^\s*Phase 2 self-accuracy: (.*)$"),
    "Statistical test (P2 self vs P1)": re.compile(r"^\s*Statistical test \(P2 self vs P1\): (.*)$")
}

def extract_log_file_metrics(log_filepath):
    extracted_log_metrics = {key: "Not found" for key in LOG_METRICS_TO_EXTRACT}
    try:
        with open(log_filepath, 'r') as f:
            for line in f:
                for metric_name, pattern in LOG_METRIC_PATTERNS.items():
                    match = pattern.match(line)
                    if match:
                        extracted_log_metrics[metric_name] = match.group(1).strip()
                        if all(val != "Not found" for val in extracted_log_metrics.values()):
                            return extracted_log_metrics
    except FileNotFoundError:
        print(f"Warning: Log file not found: {log_filepath}")
    except Exception as e:
        print(f"An error occurred while reading log file {log_filepath}: {e}")
    return extracted_log_metrics


def compare_predictors_of_answer(stated_confs, implicit_confs, pass_decisions):
    """Compare which confidence measure better predicts passing."""
    mask = ~(np.isnan(stated_confs) | np.isnan(implicit_confs) | np.isnan(pass_decisions))
    stated_confs = stated_confs[mask]
    implicit_confs = implicit_confs[mask]
    pass_decisions = pass_decisions[mask]
    #stated_ranks = rankdata(stated_confs) / len(stated_confs)
    #implicit_ranks = rankdata(implicit_confs) / len(implicit_confs)
    #X_stated = stated_ranks.reshape(-1, 1)
    #X_implicit = implicit_ranks.reshape(-1, 1)          
    X_stated = np.log((stated_confs+1e-6)/(1-stated_confs+1e-6)).reshape(-1, 1)
    eps = 1e-6 
    p     = np.clip(implicit_confs.astype(float), eps, 1 - eps)
    X_implicit = np.log(p / (1 - p)).reshape(-1, 1) 

    #X_stated = stated_confs.reshape(-1, 1)
    #X_implicit = implicit_confs.reshape(-1, 1)
    X_both = np.column_stack([stated_confs, implicit_confs])
    y = pass_decisions
    
    # Fit three models
    lr_stated = LogisticRegression().fit(X_stated, y)
    lr_implicit = LogisticRegression().fit(X_implicit, y)  
    lr_both = LogisticRegression().fit(X_both, y)
    
    # Cross-validated AUC scores
    auc_stated = cross_val_score(LogisticRegression(), X_stated, y, 
                                cv=5, scoring='roc_auc').mean()
    auc_implicit = cross_val_score(LogisticRegression(), X_implicit, y,
                                    cv=5, scoring='roc_auc').mean()
    auc_both = cross_val_score(LogisticRegression(), X_both, y,
                                cv=5, scoring='roc_auc').mean()
    
    # Get log-likelihoods 
    from sklearn.metrics import log_loss
    ll_stated = -log_loss(y, lr_stated.predict_proba(X_stated)[:,1], normalize=False)
    ll_implicit = -log_loss(y, lr_implicit.predict_proba(X_implicit)[:,1], normalize=False)
    ll_both = -log_loss(y, lr_both.predict_proba(X_both)[:,1], normalize=False)
    
    # Likelihood ratio test: does implicit add to stated?
    lr_stat_implicit_adds = 2 * (ll_both - ll_stated)
    p_value_implicit_adds = 1 - chi2.cdf(lr_stat_implicit_adds, df=1)
    
    # Likelihood ratio test: does stated add to implicit?
    lr_stat_stated_adds = 2 * (ll_both - ll_implicit)
    p_value_stated_adds = 1 - chi2.cdf(lr_stat_stated_adds, df=1)
    
    # Get standardized coefficients for interpretation
    X_both_std = (X_both - X_both.mean(axis=0)) / X_both.std(axis=0)
    lr_std = LogisticRegression().fit(X_both_std, y)
    
    results = {
        'auc_stated': auc_stated,
        'auc_implicit': auc_implicit,
        'auc_both': auc_both,
        'coef_stated': lr_std.coef_[0][0],
        'coef_implicit': lr_std.coef_[0][1],
        'p_implicit_adds_to_stated': p_value_implicit_adds,
        'p_stated_adds_to_implicit': p_value_stated_adds,
    }
    
    return results

def compare_predictors_of_implicit_conf(stated, behavior, implicit_confs):
    mask = ~(np.isnan(stated) | np.isnan(implicit_confs) | np.isnan(behavior))
    stated = stated[mask]
    implicit_confs = implicit_confs[mask]
    behavior = behavior[mask]
    eps = 1e-6 
    implicit_confs = np.clip(implicit_confs.astype(float), eps, 1 - eps)
    corr_actual, p_actual = pointbiserialr(behavior, implicit_confs)
    if stated.dtype == np.dtype('int'):
        corr_stated, p_stated = pointbiserialr(stated, implicit_confs)
    else:
        corr_stated, p_stated = pearsonr(stated, implicit_confs)
    # Test if correlations are significantly different using Fisher's z-transformation
    z_actual = np.arctanh(corr_actual)
    z_stated = np.arctanh(corr_stated)
    z_diff = (z_actual - z_stated) / np.sqrt(2/(len(implicit_confs)-3))
    p_diff = 2*(1 - norm.cdf(abs(z_diff)))
    return {
        'corr_actual': corr_actual,
        'p_actual': p_actual,
        'corr_stated': corr_stated,
        'p_stated': p_stated,
        'p_diff': p_diff
    }
