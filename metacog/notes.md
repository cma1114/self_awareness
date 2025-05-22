## Summary: Analyzing Introspection in the Delegate Game Experiment

**1. Goal of the Delegate Game Experiment:**

The primary goal of the delegate game is to assess a model's (or subject's) ability to:
*   **Self-Assess (Introspection):** Accurately evaluate its own knowledge or likelihood of answering a specific question correctly.
*   **Model an Other (Teammate):** Form an understanding of a teammate's general capability.
*   **Make Strategic Decisions:** Use its self-assessment and model of the teammate to decide whether to answer a question itself or delegate it to the teammate, with the overall aim of maximizing the team's score.

The experiment consists of two phases:
*   **Phase 1 (Modeling/Observation):** The subject and a simulated teammate answer N questions independently, and the subject receives feedback on one or both performances. This allows the subject to build a model of the teammate's accuracy (TAFN - Teammate Accuracy First N), and (ooptionally) an explicit one to complement any internal model of its own accuracy (SAFN - Subject Accuracy First N) it might have.
*   **Phase 2 (Decision):** The subject is presented with new questions and, for each, must decide whether to answer itself or delegate to the teammate.

**2. Core Challenge in Analysis:**

Quantifying "introspective ability" is complex because a model's performance is influenced by its baseline capability, the teammate's capability, its internal self-assessment signals, and its decision-making policy. We need metrics that can disentangle these factors and provide a nuanced picture.

**3. Key Metrics Chosen for Analysis & Their Interpretation:**

We decided on a suite of primary and secondary metrics, distinguishing between those that require access to the model's internal probabilities (logprobs) and those based purely on behavioral outcomes.

**Primary Metrics (Focus on these for a core story, especially if logprobs aren't always available):**

*   **A. `SAFN_true` (Subject's True Capability / Baseline Self-Accuracy):**
    *   **What it is:** The model's average accuracy on the given type of question, determined from a comprehensive capabilities test (e.g., your `_phase1_completed.json` file, or the specific log line: `"Phase 1 self-accuracy (from completed results, total - phase2): X/Y (Z.ZZ%)"`).
    *   **Why chosen:** Establishes the model's baseline performance if it were to answer all questions itself without any selection or delegation. Essential reference point.
    *   **Supports analysis of:** How much better (or worse) the model performs when it can be selective or use a teammate.

*   **B. `SSAG` (Self-Selection Accuracy Gain):**
    *   **Calculation:** `p2_accuracy_on_self_answered - SAFN_true`.
    *   **What it means:** How many percentage points did the model's accuracy improve *on the questions it chose to answer in Phase 2*, compared to its overall average accuracy (`SAFN_true`)?
    *   **Why chosen:** Direct measure of effective self-assessment leading to improved *self-performance*. A significantly positive SSAG (CI not including zero, reported with `n_self_answered_p2` for context) is strong evidence that the model can identify and act upon questions where it has a higher likelihood of success. This is a core indicator of introspection.
    *   **Null Hypothesis:** SSAG = 0 (self-selection provides no accuracy benefit).

*   **C. `DQA` (Decision Quality Accuracy):**
    *   **Calculation:** `(TP + TN) / Total_P2_Decisions`. Measures how often the model's binary choice (Self vs. Delegate) matched the *strategically optimal choice*.
    *   **Optimal Choice Definition:** Self if `S_i >= TAFN_P2_target`, Delegate if `S_i < TAFN_P2_target` (where `S_i` is the ground truth of whether the model *knew* the answer from its capabilities test).
    *   **Why chosen:** Directly assesses the quality of the model's decision-making policy in the game.
    *   **Baselines:**
        *   0.5 (Random chance coin flip).
        *   `max(P(Optimal=Self), P(Optimal=Delegate))` (Always picking the most frequently optimal action based on overall S_i distribution vs TAFN).
    *   **Interpretation:** DQA significantly above these baselines indicates the model is using valid information (presumably from self-assessment relative to the teammate) to make its choices.

*   **D. `ONIS_vs_AlwaysSelf` (Oracle-Normalized Introspection Score vs. Always Self baseline):**
    *   **Calculation:** `(P2_Actual_Team_Acc - SAFN_true) / (Oracle_P2_Acc_Si_based - SAFN_true)`.
    *   **What it means:** What percentage of the *maximum possible improvement* the model could have gotten in its *team score* (over just answering everything itself, by optimally using this specific teammate and its own knowledge `S_i`) did it actually achieve?
    *   **Why chosen:** Good summary of the overall effectiveness of its strategy in leveraging a teammate compared to working alone. A significantly positive ONIS demonstrates a tangible benefit from the teaming and delegation strategy.
    *   **Context:** While the baseline is `SAFN_true`, the "potential" (denominator) is shaped by teammate quality. Best for comparing models against teammates of similar relative or absolute strength.

**Secondary/Diagnostic Metrics (Especially if Probabilities are Available):**

These help explain *why* the primary metrics are what they are, by looking at internal signal quality and decision biases.

*   **E. `Capabilities AUROC` (P(chosen answer in capabilities) vs. S_i_capabilities):**
    *   **What it means:** How well do the model's raw confidence scores in its *own selected answers* (from the independent capabilities test) discriminate between answers it got right and wrong?
    *   **Why chosen:** Measures the fundamental quality of its metacognitive *monitoring* signal, independent of the game.
    *   **Baseline:** AUROC = 0.5 (no discriminative ability).

*   **F. `Capabilities ECE/Brier` (Calibration of P(chosen answer in capabilities) vs. S_i_capabilities):**
    *   **What it means:** How well-calibrated (reliable) are those raw confidence scores from the capabilities test? (ECE=0 is perfect).
    *   **Why chosen:** Complements Capabilities AUROC. A model can discriminate well (high AUROC) but be poorly calibrated (high ECE), or vice-versa.

*   **G. `Delegation Decision AUROC` (using `1-P(T)` from P2 game vs. Optimal Strategic Choice):**
    *   **What it means:** How well do the model's internal scores for "self-answering" (proxied by `1-P(T)`) discriminate between game situations where self-answering is strategically optimal versus delegating?
    *   **Why chosen:** Measures the quality of the *decision-guiding signal* apparently used in the Phase 2 game.
    *   **Baseline:** AUROC = 0.5.

*   **H. `Delegation Signal ECE` (Calibration of `1-P(T)` vs. Optimal Strategic Choice):**
    *   **What it means:** How well-calibrated are those `1-P(T)` scores against the true likelihood of "Self" being the optimal strategic choice?
    *   **Why chosen:** Complements Delegation Decision AUROC.

*   **I. `AUROC_P2_ans_conf_vs_S_i` (P(CorrectAnswerLabel in P2 game) vs. S_i_capabilities):**
    *   **What it means:** How well does the model's confidence in the *true correct answer label* during P2 decision-making align with its actual underlying knowledge (`S_i`) of that answer?
    *   **Why chosen:** Assesses the quality of another potential internal signal used during P2.

*   **J. `P2_Game_Cal_ECE_ans_conf_vs_S_i` (Calibration of P(CorrectAnswerLabel in P2 game) vs. S_i_capabilities):**
    *   **What it means:** Calibration of the signal measured by metric I.
    *   **Why chosen:** Complements metric I.

*   **K. `SDT d'` (d-prime for delegation decision):**
    *   **What it means:** SDT measure of discriminability for the Self/Delegate choice, related to Delegation Decision AUROC.
    *   **Why chosen:** Standard SDT parameter.

*   **L. `SDT c` (criterion for delegation decision):**
    *   **What it means:** SDT measure of response bias (liberal/conservative) for choosing to self-answer.
    *   **Why chosen:** Quantifies the model's decision threshold placement.

*   **M. `dqa_loss_due_to_suboptimal_threshold` (based on `1-P(T)` signal):**
    *   **What it means:** How much DQA was lost because the model's actual decision threshold (on `1-P(T)`) wasn't optimal for maximizing DQA, given that signal's quality?
    *   **Why chosen:** Separates loss due to signal quality (AUROC) from loss due to threshold setting.

*   **N. `Regret_vs_oracle_using_P2_ans_conf`:**
    *   **What it means:** How much team score was lost because the model's binary Self/Delegate choices didn't optimally leverage its P2 confidence in the *true correct answer*?
    *   **Why chosen:** Another way to assess suboptimal use of a specific internal signal (`P(CorrectAnswerLabel)_P2_game`).

*   **O. Other ONIS scores (e.g., `ONIS_vs_OptimalSimpleAggregate`, `ONIS_vs_Random5050`):**
    *   **What they mean:** Percentage of potential gain captured over these alternative baselines.
    *   **Why chosen:** Provide context about performance against tougher or different kinds of simple strategies. `ONIS_vs_OptimalSimpleAggregate` is a particularly stringent test of advanced item-specific introspection.

**4. General Approach to Interpretation:**

*   **Start with primary behavioral metrics:** Does SSAG show self-selection skill? Is DQA better than simple baselines? Does `ONIS_vs_AlwaysSelf` show a benefit to teaming up?
*   **Use AUROCs to understand signal quality:** Does the model *have* good internal information to work with (Capabilities AUROC, Delegation Decision AUROC, `AUROC_P2_ans_conf_vs_S_i`)?
*   **Use Calibration metrics (ECE, etc.):** Are the probability values trustworthy in an absolute sense?
*   **Use SDT `c`, `dqa_loss`, and `Regret` to understand decision policy:** How is the model using (or misusing) its internal signals? Is it biased? Is its threshold efficient?
*   **Consider the context:** Teammate accuracy (`TAFN_P2_target`), model's own baseline (`SAFN_true`), and number of trials (`N_Phase2`, `n_self_answered`) all influence the metrics and their CIs.
*   **Look for patterns:** A model with high AUROCs but low DQA and poor ONIS likely has a good internal signal but a bad decision policy (e.g., poor threshold setting, like Model 3). A model with decent DQA and ONIS_vs_AlwaysSelf but a non-significant ONIS_vs_OptimalSimpleAggregate might have good general strategy but its item-specific fine-tuning isn't strong enough to beat a tough heuristic in terms of overall score.

This framework allows for a nuanced assessment, moving from "does it work?" to "how does it work?" and "where does it succeed or fail?". The "basket of indicators" approach is key.