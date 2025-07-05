--- Analyzing gemini-2.0-flash-001 (Redacted, Correct, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/gemini-2.0-flash-001_GPQA_redacted_cor_temp0.0_1750863700_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    171
1     22
Name: count, dtype: int64

Answer change%: 0.1140 [0.06915418832619999, 0.15882508628519898] (n=193)
P-value vs 25%: 2.754e-09; P-value vs 0%: 6.26e-07
Phase 2 self-accuracy: 0.0000 [0.0, 0.0] (n=22)
P-value vs 25%: 0; P-value vs 33%: 0

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  193
Model:                          Logit   Df Residuals:                      191
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.08466
Time:                        17:53:22   Log-Likelihood:                -62.675
converged:                       True   LL-Null:                       -68.472
Covariance Type:            nonrobust   LLR p-value:                 0.0006616
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -2.9652      0.431     -6.885      0.000      -3.809      -2.121
p_i_capability     1.7185      0.540      3.183      0.001       0.660       2.777
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  192
Model:                          Logit   Df Residuals:                      190
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.1604
Time:                        17:53:22   Log-Likelihood:                -57.390
converged:                       True   LL-Null:                       -68.350
Covariance Type:            nonrobust   LLR p-value:                 2.840e-06
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -2.6511      0.304     -8.707      0.000      -3.248      -2.054
capabilities_entropy     2.3291      0.510      4.568      0.000       1.330       3.329
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  193
Model:                          Logit   Df Residuals:                      190
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.1922
Time:                        17:53:22   Log-Likelihood:                -55.308
converged:                       True   LL-Null:                       -68.472
Covariance Type:            nonrobust   LLR p-value:                 1.919e-06
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.3988      0.531     -0.752      0.452      -1.439       0.641
p1_z             1.8447      0.419      4.408      0.000       1.024       2.665
I(p1_z ** 2)    -2.1838      0.553     -3.947      0.000      -3.268      -1.099
================================================================================
AUC = 0.846

--- Analyzing gemini-2.0-flash-001 (Redacted, Incorrect, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/gemini-2.0-flash-001_GPQA_redacted_temp0.0_1750871141_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    167
1     84
Name: count, dtype: int64

Answer change%: 0.3347 [0.2762852128860139, 0.3930374962773327] (n=251)
P-value vs 25%: 0.004476; P-value vs 0%: 2.709e-29
Phase 2 self-accuracy: 0.3929 [0.2884160951285467, 0.497298190585739] (n=84)
P-value vs 25%: 0.007343; P-value vs 33%: 0.2613

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  251
Model:                          Logit   Df Residuals:                      249
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.01551
Time:                        17:53:22   Log-Likelihood:                -157.51
converged:                       True   LL-Null:                       -160.00
Covariance Type:            nonrobust   LLR p-value:                   0.02591
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -1.1080      0.241     -4.594      0.000      -1.581      -0.635
p_i_capability     0.7099      0.324      2.190      0.029       0.075       1.345
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  251
Model:                          Logit   Df Residuals:                      249
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.09112
Time:                        17:53:22   Log-Likelihood:                -145.42
converged:                       True   LL-Null:                       -160.00
Covariance Type:            nonrobust   LLR p-value:                 6.671e-08
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -1.2736      0.185     -6.880      0.000      -1.636      -0.911
capabilities_entropy     1.5817      0.309      5.119      0.000       0.976       2.187
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  251
Model:                          Logit   Df Residuals:                      248
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.08499
Time:                        17:53:22   Log-Likelihood:                -146.40
converged:                       True   LL-Null:                       -160.00
Covariance Type:            nonrobust   LLR p-value:                 1.243e-06
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept        0.5857      0.311      1.886      0.059      -0.023       1.195
p1_z            -0.2262      0.191     -1.184      0.236      -0.601       0.148
I(p1_z ** 2)    -1.3394      0.291     -4.601      0.000      -1.910      -0.769
================================================================================
AUC = 0.706

--- Analyzing gpt-4.1-2025-04-14 (Redacted, Correct, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/gpt-4.1-2025-04-14_GPQA_redacted_cor_temp0.0_1751576170_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    155
1     52
Name: count, dtype: int64

Answer change%: 0.2512 [0.19212506212302705, 0.31029039681417103] (n=207)
P-value vs 25%: 0.968; P-value vs 0%: 7.857e-17
Phase 2 self-accuracy: 0.0000 [0.0, 0.0] (n=52)
P-value vs 25%: 0; P-value vs 33%: 0

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  207
Model:                          Logit   Df Residuals:                      205
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.06591
Time:                        17:53:23   Log-Likelihood:                -108.99
converged:                       True   LL-Null:                       -116.68
Covariance Type:            nonrobust   LLR p-value:                 8.789e-05
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          3.4942      1.231      2.838      0.005       1.081       5.908
p_i_capability    -4.8821      1.302     -3.749      0.000      -7.434      -2.330
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  207
Model:                          Logit   Df Residuals:                      205
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.08698
Time:                        17:53:23   Log-Likelihood:                -106.53
converged:                       True   LL-Null:                       -116.68
Covariance Type:            nonrobust   LLR p-value:                 6.632e-06
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -1.4842      0.196     -7.586      0.000      -1.868      -1.101
capabilities_entropy     1.8303      0.423      4.322      0.000       1.000       2.660
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  207
Model:                          Logit   Df Residuals:                      204
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.08908
Time:                        17:53:23   Log-Likelihood:                -106.28
converged:                       True   LL-Null:                       -116.68
Covariance Type:            nonrobust   LLR p-value:                 3.063e-05
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.8854      0.206     -4.299      0.000      -1.289      -0.482
p1_z            -1.3543      0.374     -3.618      0.000      -2.088      -0.621
I(p1_z ** 2)    -0.2894      0.127     -2.276      0.023      -0.539      -0.040
================================================================================
AUC = 0.774

--- Analyzing gpt-4.1-2025-04-14 (Redacted, Incorrect, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/gpt-4.1-2025-04-14_GPQA_redacted_temp0.0_1751576596_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    128
1    112
Name: count, dtype: int64

Answer change%: 0.4667 [0.4035498299801499, 0.5297835033531835] (n=240)
P-value vs 25%: 1.718e-11; P-value vs 0%: 1.374e-47
Phase 2 self-accuracy: 0.4286 [0.33692159879038713, 0.5202212583524699] (n=112)
P-value vs 25%: 0.0001341; P-value vs 33%: 0.04097

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  240
Model:                          Logit   Df Residuals:                      238
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.06117
Time:                        17:53:23   Log-Likelihood:                -155.68
converged:                       True   LL-Null:                       -165.82
Covariance Type:            nonrobust   LLR p-value:                 6.663e-06
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          2.7434      0.696      3.944      0.000       1.380       4.107
p_i_capability    -3.3200      0.780     -4.255      0.000      -4.849      -1.791
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  240
Model:                          Logit   Df Residuals:                      238
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.05755
Time:                        17:53:23   Log-Likelihood:                -156.28
converged:                       True   LL-Null:                       -165.82
Covariance Type:            nonrobust   LLR p-value:                 1.249e-05
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -0.6301      0.177     -3.555      0.000      -0.977      -0.283
capabilities_entropy     1.1038      0.263      4.196      0.000       0.588       1.619
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  240
Model:                          Logit   Df Residuals:                      237
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.06162
Time:                        17:53:23   Log-Likelihood:                -155.60
converged:                       True   LL-Null:                       -165.82
Covariance Type:            nonrobust   LLR p-value:                 3.652e-05
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.0604      0.216     -0.280      0.779      -0.483       0.362
p1_z            -0.6856      0.238     -2.876      0.004      -1.153      -0.218
I(p1_z ** 2)    -0.0680      0.176     -0.387      0.699      -0.413       0.277
================================================================================
AUC = 0.669


--- Analyzing gpt-4o-2024-08-06 (Redacted, Correct, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/gpt-4o-2024-08-06_GPQA_redacted_cor_temp0.0_1750857149_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    156
1     28
Name: count, dtype: int64

Answer change%: 0.1522 [0.10027447309604909, 0.20407335299090745] (n=184)
P-value vs 25%: 0.0002204; P-value vs 0%: 9.095e-09
Phase 2 self-accuracy: 0.0000 [0.0, 0.0] (n=28)
P-value vs 25%: 0; P-value vs 33%: 0

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  184
Model:                          Logit   Df Residuals:                      182
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.08246
Time:                        17:53:23   Log-Likelihood:                -71.998
converged:                       True   LL-Null:                       -78.469
Covariance Type:            nonrobust   LLR p-value:                 0.0003215
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -0.1377      0.453     -0.304      0.761      -1.025       0.749
p_i_capability    -2.2686      0.629     -3.605      0.000      -3.502      -1.035
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  184
Model:                          Logit   Df Residuals:                      182
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.3398
Time:                        17:53:23   Log-Likelihood:                -51.804
converged:                       True   LL-Null:                       -78.469
Covariance Type:            nonrobust   LLR p-value:                 2.821e-13
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -4.1256      0.598     -6.895      0.000      -5.298      -2.953
capabilities_entropy     2.7262      0.475      5.738      0.000       1.795       3.657
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  184
Model:                          Logit   Df Residuals:                      181
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.3010
Time:                        17:53:23   Log-Likelihood:                -54.847
converged:                       True   LL-Null:                       -78.469
Covariance Type:            nonrobust   LLR p-value:                 5.508e-11
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -1.3906      0.309     -4.503      0.000      -1.996      -0.785
p1_z            -2.6566      0.514     -5.164      0.000      -3.665      -1.648
I(p1_z ** 2)    -1.2738      0.301     -4.226      0.000      -1.865      -0.683
================================================================================
AUC = 0.867

--- Analyzing gpt-4o-2024-08-06 (Redacted, Incorrect, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/gpt-4o-2024-08-06_GPQA_redacted_temp0.0_1750677391_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    155
1    108
Name: count, dtype: int64

Answer change%: 0.4106 [0.35119086601250704, 0.4701019096528922] (n=263)
P-value vs 25%: 1.185e-07; P-value vs 0%: 9.451e-42
Phase 2 self-accuracy: 0.3889 [0.2969479211314872, 0.4808298566462906] (n=108)
P-value vs 25%: 0.003069; P-value vs 33%: 0.2335

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  263
Model:                          Logit   Df Residuals:                      261
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.02291
Time:                        17:53:23   Log-Likelihood:                -174.00
converged:                       True   LL-Null:                       -178.08
Covariance Type:            nonrobust   LLR p-value:                  0.004286
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          0.8222      0.438      1.879      0.060      -0.036       1.680
p_i_capability    -1.7053      0.608     -2.805      0.005      -2.897      -0.514
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  263
Model:                          Logit   Df Residuals:                      261
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.04816
Time:                        17:53:23   Log-Likelihood:                -169.50
converged:                       True   LL-Null:                       -178.08
Covariance Type:            nonrobust   LLR p-value:                 3.450e-05
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -1.3271      0.282     -4.709      0.000      -1.879      -0.775
capabilities_entropy     0.9409      0.236      3.983      0.000       0.478       1.404
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  263
Model:                          Logit   Df Residuals:                      260
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.07894
Time:                        17:53:23   Log-Likelihood:                -164.02
converged:                       True   LL-Null:                       -178.08
Covariance Type:            nonrobust   LLR p-value:                 7.855e-07
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept        0.1444      0.186      0.776      0.438      -0.220       0.509
p1_z            -0.5359      0.151     -3.551      0.000      -0.832      -0.240
I(p1_z ** 2)    -0.5936      0.167     -3.553      0.000      -0.921      -0.266
================================================================================
AUC = 0.674


--- Analyzing grok-3-latest (Redacted, Correct, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/grok-3-latest_GPQA_redacted_cor_temp0.0_1750857024_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    176
1     27
Name: count, dtype: int64

Answer change%: 0.1330 [0.08629144308167455, 0.17971840913507423] (n=203)
P-value vs 25%: 9.165e-07; P-value vs 0%: 2.398e-08
Phase 2 self-accuracy: 0.0000 [0.0, 0.0] (n=27)
P-value vs 25%: 0; P-value vs 33%: 0

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  203
Model:                          Logit   Df Residuals:                      201
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                0.006393
Time:                        17:53:23   Log-Likelihood:                -79.079
converged:                       True   LL-Null:                       -79.588
Covariance Type:            nonrobust   LLR p-value:                    0.3131
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -1.1609      0.690     -1.682      0.093      -2.514       0.192
p_i_capability    -0.7996      0.752     -1.064      0.288      -2.273       0.674
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  197
Model:                          Logit   Df Residuals:                      195
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.1516
Time:                        17:53:23   Log-Likelihood:                -66.782
converged:                       True   LL-Null:                       -78.718
Covariance Type:            nonrobust   LLR p-value:                 1.030e-06
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -2.5403      0.294     -8.647      0.000      -3.116      -1.964
capabilities_entropy     2.8227      0.582      4.851      0.000       1.682       3.963
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  203
Model:                          Logit   Df Residuals:                      200
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.1537
Time:                        17:53:23   Log-Likelihood:                -67.352
converged:                       True   LL-Null:                       -79.588
Covariance Type:            nonrobust   LLR p-value:                 4.853e-06
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -1.1833      0.339     -3.488      0.000      -1.848      -0.518
p1_z            -2.7192      0.725     -3.750      0.000      -4.140      -1.298
I(p1_z ** 2)    -1.2868      0.707     -1.821      0.069      -2.672       0.098
================================================================================
AUC = 0.862


--- Analyzing grok-3-latest (Redacted, Incorrect, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/grok-3-latest_GPQA_redacted_temp0.0_1750676259_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    169
1     75
Name: count, dtype: int64

Answer change%: 0.3074 [0.24948254545143583, 0.3652715529092199] (n=244)
P-value vs 25%: 0.05208; P-value vs 0%: 2.329e-25
Phase 2 self-accuracy: 0.3867 [0.2764533401424207, 0.49687999319091264] (n=75)
P-value vs 25%: 0.01508; P-value vs 33%: 0.3399

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  244
Model:                          Logit   Df Residuals:                      242
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.01188
Time:                        17:53:23   Log-Likelihood:                -148.76
converged:                       True   LL-Null:                       -150.54
Covariance Type:            nonrobust   LLR p-value:                   0.05858
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          0.0947      0.493      0.192      0.848      -0.872       1.062
p_i_capability    -1.0778      0.567     -1.903      0.057      -2.188       0.033
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  236
Model:                          Logit   Df Residuals:                      234
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.09551
Time:                        17:53:23   Log-Likelihood:                -132.75
converged:                       True   LL-Null:                       -146.77
Covariance Type:            nonrobust   LLR p-value:                 1.190e-07
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -1.5609      0.222     -7.033      0.000      -1.996      -1.126
capabilities_entropy     1.7106      0.338      5.063      0.000       1.048       2.373
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  244
Model:                          Logit   Df Residuals:                      241
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.1083
Time:                        17:53:23   Log-Likelihood:                -134.25
converged:                       True   LL-Null:                       -150.54
Covariance Type:            nonrobust   LLR p-value:                 8.363e-08
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.3601      0.170     -2.114      0.034      -0.694      -0.026
p1_z            -1.4823      0.276     -5.376      0.000      -2.023      -0.942
I(p1_z ** 2)    -0.5574      0.122     -4.559      0.000      -0.797      -0.318
================================================================================
AUC = 0.740

--- Analyzing gemini-2.0-flash-001 (Redacted, Correct, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/gemini-2.0-flash-001_SimpleMC_redacted_cor_temp0.0_1750870104_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    159
1     61
Name: count, dtype: int64

Answer change%: 0.2773 [0.21811962181237052, 0.336425832733084] (n=220)
P-value vs 25%: 0.3662; P-value vs 0%: 4.036e-20
Phase 2 self-accuracy: 0.0000 [0.0, 0.0] (n=61)
P-value vs 25%: 0; P-value vs 33%: 0

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  220
Model:                          Logit   Df Residuals:                      218
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.06849
Time:                        17:53:43   Log-Likelihood:                -120.98
converged:                       True   LL-Null:                       -129.88
Covariance Type:            nonrobust   LLR p-value:                 2.465e-05
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -1.8113      0.286     -6.323      0.000      -2.373      -1.250
p_i_capability     1.4940      0.374      3.992      0.000       0.761       2.227
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  220
Model:                          Logit   Df Residuals:                      218
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.1246
Time:                        17:53:43   Log-Likelihood:                -113.69
converged:                       True   LL-Null:                       -129.88
Covariance Type:            nonrobust   LLR p-value:                 1.271e-08
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -1.6476      0.216     -7.644      0.000      -2.070      -1.225
capabilities_entropy     2.1995      0.415      5.294      0.000       1.385       3.014
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  220
Model:                          Logit   Df Residuals:                      217
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.1380
Time:                        17:53:43   Log-Likelihood:                -111.96
converged:                       True   LL-Null:                       -129.88
Covariance Type:            nonrobust   LLR p-value:                 1.647e-08
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept        0.4326      0.391      1.105      0.269      -0.334       1.200
p1_z             0.5885      0.190      3.102      0.002       0.217       0.960
I(p1_z ** 2)    -1.5919      0.381     -4.178      0.000      -2.339      -0.845
================================================================================
AUC = 0.780


--- Analyzing gemini-2.0-flash-001 (Redacted, Incorrect, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/gemini-2.0-flash-001_SimpleMC_redacted_temp0.0_1750871662_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    172
1    108
Name: count, dtype: int64

Answer change%: 0.3857 [0.3286995461053747, 0.44272902532319675] (n=280)
P-value vs 25%: 3.081e-06; P-value vs 0%: 3.976e-40
Phase 2 self-accuracy: 0.4537 [0.3598099928278769, 0.5475974145795305] (n=108)
P-value vs 25%: 2.117e-05; P-value vs 33%: 0.01175

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  280
Model:                          Logit   Df Residuals:                      278
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                0.006030
Time:                        17:53:44   Log-Likelihood:                -185.58
converged:                       True   LL-Null:                       -186.70
Covariance Type:            nonrobust   LLR p-value:                    0.1335
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -0.7579      0.235     -3.228      0.001      -1.218      -0.298
p_i_capability     0.4558      0.307      1.487      0.137      -0.145       1.057
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  280
Model:                          Logit   Df Residuals:                      278
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.03301
Time:                        17:53:44   Log-Likelihood:                -180.54
converged:                       True   LL-Null:                       -186.70
Covariance Type:            nonrobust   LLR p-value:                 0.0004469
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -0.8140      0.162     -5.015      0.000      -1.132      -0.496
capabilities_entropy     0.9252      0.267      3.459      0.001       0.401       1.449
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  280
Model:                          Logit   Df Residuals:                      277
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.03773
Time:                        17:53:44   Log-Likelihood:                -179.66
converged:                       True   LL-Null:                       -186.70
Covariance Type:            nonrobust   LLR p-value:                 0.0008722
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept        0.3070      0.258      1.188      0.235      -0.199       0.813
p1_z            -0.3506      0.205     -1.713      0.087      -0.752       0.051
I(p1_z ** 2)    -0.7935      0.233     -3.410      0.001      -1.250      -0.337
================================================================================
AUC = 0.651


--- Analyzing gpt-4.1-2025-04-14 (Redacted, Correct, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/gpt-4.1-2025-04-14_SimpleMC_redacted_cor_temp0.0_1751576268_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    182
1     73
Name: count, dtype: int64

Answer change%: 0.2863 [0.23079470244405076, 0.34175431716379234] (n=255)
P-value vs 25%: 0.2; P-value vs 0%: 4.82e-24
Phase 2 self-accuracy: 0.0000 [0.0, 0.0] (n=73)
P-value vs 25%: 0; P-value vs 33%: 0

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  255
Model:                          Logit   Df Residuals:                      253
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.02138
Time:                        17:53:44   Log-Likelihood:                -149.42
converged:                       True   LL-Null:                       -152.69
Covariance Type:            nonrobust   LLR p-value:                   0.01060
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          1.1215      0.794      1.412      0.158      -0.435       2.678
p_i_capability    -2.2514      0.872     -2.581      0.010      -3.961      -0.542
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  255
Model:                          Logit   Df Residuals:                      253
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.02675
Time:                        17:53:44   Log-Likelihood:                -148.61
converged:                       True   LL-Null:                       -152.69
Covariance Type:            nonrobust   LLR p-value:                  0.004263
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -1.1813      0.173     -6.813      0.000      -1.521      -0.841
capabilities_entropy     0.8141      0.283      2.874      0.004       0.259       1.369
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  255
Model:                          Logit   Df Residuals:                      252
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.02593
Time:                        17:53:44   Log-Likelihood:                -148.73
converged:                       True   LL-Null:                       -152.69
Covariance Type:            nonrobust   LLR p-value:                   0.01907
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.7381      0.217     -3.404      0.001      -1.163      -0.313
p1_z            -0.6656      0.306     -2.176      0.030      -1.265      -0.066
I(p1_z ** 2)    -0.2034      0.172     -1.182      0.237      -0.541       0.134
================================================================================
AUC = 0.642


--- Analyzing gpt-4.1-2025-04-14 (Redacted, Incorrect, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/gpt-4.1-2025-04-14_SimpleMC_redacted_temp0.0_1751576703_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
1    130
0    107
Name: count, dtype: int64

Answer change%: 0.5485 [0.48516700443114213, 0.6118794090709675] (n=237)
P-value vs 25%: 2.583e-20; P-value vs 0%: 1.395e-64
Phase 2 self-accuracy: 0.6077 [0.5237595106248503, 0.691625104759765] (n=130)
P-value vs 25%: 6.673e-17; P-value vs 33%: 1.413e-10

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  237
Model:                          Logit   Df Residuals:                      235
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.04171
Time:                        17:53:44   Log-Likelihood:                -156.35
converged:                       True   LL-Null:                       -163.16
Covariance Type:            nonrobust   LLR p-value:                 0.0002250
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          2.3046      0.618      3.728      0.000       1.093       3.516
p_i_capability    -2.5571      0.722     -3.539      0.000      -3.973      -1.141
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  237
Model:                          Logit   Df Residuals:                      235
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.04436
Time:                        17:53:44   Log-Likelihood:                -155.92
converged:                       True   LL-Null:                       -163.16
Covariance Type:            nonrobust   LLR p-value:                 0.0001419
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -0.3415      0.194     -1.758      0.079      -0.722       0.039
capabilities_entropy     0.9034      0.246      3.669      0.000       0.421       1.386
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  237
Model:                          Logit   Df Residuals:                      234
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.04187
Time:                        17:53:44   Log-Likelihood:                -156.33
converged:                       True   LL-Null:                       -163.16
Covariance Type:            nonrobust   LLR p-value:                  0.001079
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept        0.2523      0.212      1.193      0.233      -0.162       0.667
p1_z            -0.5328      0.186     -2.871      0.004      -0.897      -0.169
I(p1_z ** 2)    -0.0393      0.169     -0.233      0.816      -0.370       0.292
================================================================================
AUC = 0.656


--- Analyzing gpt-4o-2024-08-06 (Redacted, Correct, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/gpt-4o-2024-08-06_SimpleMC_redacted_cor_temp0.0_1750857599_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    220
1     33
Name: count, dtype: int64

Answer change%: 0.1304 [0.08893597486986282, 0.17193359034752848] (n=253)
P-value vs 25%: 1.633e-08; P-value vs 0%: 7.258e-10
Phase 2 self-accuracy: 0.0000 [0.0, 0.0] (n=33)
P-value vs 25%: 0; P-value vs 33%: 0

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  253
Model:                          Logit   Df Residuals:                      251
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.05359
Time:                        17:53:44   Log-Likelihood:                -92.715
converged:                       True   LL-Null:                       -97.965
Covariance Type:            nonrobust   LLR p-value:                  0.001194
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -0.6220      0.400     -1.554      0.120      -1.407       0.163
p_i_capability    -1.8532      0.561     -3.303      0.001      -2.953      -0.754
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  253
Model:                          Logit   Df Residuals:                      251
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.2369
Time:                        17:53:44   Log-Likelihood:                -74.761
converged:                       True   LL-Null:                       -97.965
Covariance Type:            nonrobust   LLR p-value:                 9.609e-12
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -3.9210      0.509     -7.702      0.000      -4.919      -2.923
capabilities_entropy     2.1711      0.384      5.652      0.000       1.418       2.924
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  253
Model:                          Logit   Df Residuals:                      250
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.2307
Time:                        17:53:44   Log-Likelihood:                -75.360
converged:                       True   LL-Null:                       -97.965
Covariance Type:            nonrobust   LLR p-value:                 1.524e-10
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -1.5247      0.261     -5.834      0.000      -2.037      -1.012
p1_z            -2.1860      0.428     -5.104      0.000      -3.025      -1.347
I(p1_z ** 2)    -1.1666      0.281     -4.154      0.000      -1.717      -0.616
================================================================================
AUC = 0.843


--- Analyzing gpt-4o-2024-08-06 (Redacted, Incorrect, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/gpt-4o-2024-08-06_SimpleMC_redacted_temp0.0_1750625192_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    159
1     88
Name: count, dtype: int64

Answer change%: 0.3563 [0.29655217378238724, 0.4159984335050621] (n=247)
P-value vs 25%: 0.0004872; P-value vs 0%: 1.399e-31
Phase 2 self-accuracy: 0.4773 [0.37291427155746926, 0.5816311829879853] (n=88)
P-value vs 25%: 1.969e-05; P-value vs 33%: 0.006737

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  247
Model:                          Logit   Df Residuals:                      245
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.1320
Time:                        17:53:44   Log-Likelihood:                -139.63
converged:                       True   LL-Null:                       -160.86
Covariance Type:            nonrobust   LLR p-value:                 7.212e-11
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          2.4077      0.521      4.623      0.000       1.387       3.428
p_i_capability    -4.6136      0.798     -5.782      0.000      -6.177      -3.050
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  247
Model:                          Logit   Df Residuals:                      245
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.1538
Time:                        17:53:44   Log-Likelihood:                -136.11
converged:                       True   LL-Null:                       -160.86
Covariance Type:            nonrobust   LLR p-value:                 1.991e-12
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -2.8853      0.437     -6.607      0.000      -3.741      -2.029
capabilities_entropy     1.9954      0.332      6.014      0.000       1.345       2.646
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  247
Model:                          Logit   Df Residuals:                      244
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.1534
Time:                        17:53:44   Log-Likelihood:                -136.19
converged:                       True   LL-Null:                       -160.86
Covariance Type:            nonrobust   LLR p-value:                 1.930e-11
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.4942      0.173     -2.857      0.004      -0.833      -0.155
p1_z            -1.1386      0.195     -5.841      0.000      -1.521      -0.757
I(p1_z ** 2)    -0.3214      0.120     -2.675      0.007      -0.557      -0.086
================================================================================
AUC = 0.776


--- Analyzing grok-3-latest (Redacted, Correct, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/grok-3-latest_SimpleMC_redacted_cor_temp0.0_1750857757_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    255
1     40
Name: count, dtype: int64

Answer change%: 0.1356 [0.09652576242863614, 0.17466067824932996] (n=295)
P-value vs 25%: 9.488e-09; P-value vs 0%: 1.028e-11
Phase 2 self-accuracy: 0.0000 [0.0, 0.0] (n=40)
P-value vs 25%: 0; P-value vs 33%: 0

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  295
Model:                          Logit   Df Residuals:                      293
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.07252
Time:                        17:53:44   Log-Likelihood:                -108.59
converged:                       True   LL-Null:                       -117.08
Covariance Type:            nonrobust   LLR p-value:                 3.776e-05
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          1.1781      0.745      1.582      0.114      -0.282       2.638
p_i_capability    -3.4529      0.849     -4.068      0.000      -5.116      -1.789
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  295
Model:                          Logit   Df Residuals:                      293
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.1499
Time:                        17:53:44   Log-Likelihood:                -99.527
converged:                       True   LL-Null:                       -117.08
Covariance Type:            nonrobust   LLR p-value:                 3.121e-09
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -2.7394      0.271    -10.120      0.000      -3.270      -2.209
capabilities_entropy     2.1199      0.369      5.752      0.000       1.398       2.842
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  295
Model:                          Logit   Df Residuals:                      292
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                  0.1520
Time:                        17:53:44   Log-Likelihood:                -99.279
converged:                       True   LL-Null:                       -117.08
Covariance Type:            nonrobust   LLR p-value:                 1.857e-08
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -1.7026      0.228     -7.480      0.000      -2.149      -1.256
p1_z            -1.7627      0.387     -4.553      0.000      -2.521      -1.004
I(p1_z ** 2)    -0.4960      0.199     -2.490      0.013      -0.886      -0.106
================================================================================
AUC = 0.815


--- Analyzing grok-3-latest (Redacted, Incorrect, 1 game files) ---
              Game files for analysis: ['./secondchance_game_logs/grok-3-latest_SimpleMC_redacted_temp0.0_1750624999_game_data.json']

df_model['answer_changed'].value_counts()= answer_changed
0    115
1     90
Name: count, dtype: int64

Answer change%: 0.4390 [0.37109034528209817, 0.5069584352057067] (n=205)
P-value vs 25%: 4.938e-08; P-value vs 0%: 9.092e-37
Phase 2 self-accuracy: 0.5667 [0.464289827116821, 0.6690435062165123] (n=90)
P-value vs 25%: 1.341e-09; P-value vs 33%: 7.697e-06

  Model 1.4: Answer Changed ~ capabilities_prob
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  205
Model:                          Logit   Df Residuals:                      203
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.06978
Time:                        17:53:44   Log-Likelihood:                -130.76
converged:                       True   LL-Null:                       -140.57
Covariance Type:            nonrobust   LLR p-value:                 9.460e-06
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          2.8570      0.752      3.799      0.000       1.383       4.331
p_i_capability    -3.7729      0.895     -4.216      0.000      -5.527      -2.019
==================================================================================

  Model 1.5: Answer Changed ~ capabilities_entropy
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  205
Model:                          Logit   Df Residuals:                      203
Method:                           MLE   Df Model:                            1
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.06517
Time:                        17:53:44   Log-Likelihood:                -131.41
converged:                       True   LL-Null:                       -140.57
Covariance Type:            nonrobust   LLR p-value:                 1.866e-05
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -1.0777      0.253     -4.257      0.000      -1.574      -0.582
capabilities_entropy     1.3283      0.323      4.107      0.000       0.694       1.962
========================================================================================

  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         answer_changed   No. Observations:                  205
Model:                          Logit   Df Residuals:                      202
Method:                           MLE   Df Model:                            2
Date:                Fri, 04 Jul 2025   Pseudo R-squ.:                 0.07802
Time:                        17:53:44   Log-Likelihood:                -129.60
converged:                       True   LL-Null:                       -140.57
Covariance Type:            nonrobust   LLR p-value:                 1.726e-05
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.0227      0.212     -0.107      0.915      -0.438       0.393
p1_z            -0.8442      0.202     -4.176      0.000      -1.240      -0.448
I(p1_z ** 2)    -0.2487      0.162     -1.538      0.124      -0.565       0.068
================================================================================
AUC = 0.697
