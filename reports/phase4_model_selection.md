# Phase 4 Model Selection Protocol

## Objective

Phase 4 evaluates whether a constrained XGBoost challenger provides a material
development-set improvement over the Phase 3 class-weighted logistic baseline.
It produces a reproducible candidate decision, not a production approval.

## Temporal roles

| Period | Role in Phase 4 |
| --- | --- |
| Month 0–4 | Fit preprocessing and model parameters |
| Month 5 | Diagnostic stability and probability context only |
| Month 6 | Candidate ranking and model-selection guardrails |
| Month 7 | Reserved test; not exposed or evaluated |

The experiment container deliberately has no test features or test labels. It
stores only the number of reserved test rows as an audit check.

## Class imbalance

Every XGBoost candidate receives:

```text
scale_pos_weight = number of negative train rows / number of positive train rows
```

The ratio is calculated from month 0–4 labels only. It is not configured from
the full dataset and is not recomputed on calibration or validation.

## Fixed candidates

The candidate set is deliberately small. It is an explicit comparison of three
reasonable regularization profiles rather than an expensive grid search.

| Candidate | Trees | Rate | Depth | Child weight | Row sample | Column sample | Alpha | Lambda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `xgboost_shallow_regularized` | 450 | 0.05 | 3 | 10 | 0.80 | 0.80 | 0.10 | 5 |
| `xgboost_moderate_depth` | 400 | 0.05 | 4 | 15 | 0.80 | 0.80 | 0.10 | 7 |
| `xgboost_strong_regularization` | 325 | 0.08 | 3 | 25 | 0.90 | 0.90 | 0.25 | 10 |

All candidates use the histogram tree method, a binary logistic objective, a
fixed random seed, and fixed boosting rounds. Early stopping is intentionally
not used, so no later temporal partition is repeatedly consumed during fitting.

## Locked selection rule

1. Rank XGBoost candidates by validation average precision, descending.
2. Break candidate ties with validation recall at the 5% review capacity,
   validation Brier score, training time, then model name.
3. Compare the highest-ranked XGBoost candidate with
   `logistic_regression_balanced`.
4. Promote XGBoost only when both conditions hold:
   - Absolute validation average-precision improvement is at least `0.002`.
   - Validation recall at 5% capacity is no more than `0.01` below baseline.
5. Otherwise retain the logistic baseline.

Calibration-month performance is displayed but never used to select a model.
Test performance cannot enter the rule because test metrics are rejected by the
selection-table builder.

## Interpretation limits

- Raw XGBoost probabilities are not yet calibrated.
- The `0.50` cutoff remains diagnostic and is not an operating threshold.
- Built-in gain importance is an association and debugging aid, not a causal
  explanation or a substitute for the Phase 7 explainability analysis.
- A selected development candidate still requires Phase 5 calibration and
  threshold work before the one-time final test evaluation.
- Model selection on one temporal validation period can favor that period;
  later monitoring and stability analysis remain required.

## Reproducible evidence

After restarting and running all cells in
`notebooks/04_xgboost_model_selection.ipynb`, review:

- Split summary and train-only `scale_pos_weight`.
- Candidate configuration and training duration.
- Calibration-versus-validation comparison table.
- Validation metrics at each review capacity.
- Locked selection decision and guardrail outcomes.
- Selected model diagnostic errors and XGBoost gain importance.
- Artifact metadata confirming `test_evaluated: false`.

Machine-readable outputs are written to `artifacts/phase4/` and remain ignored
by Git. The notebook output is the committed experiment evidence after a clean
restart and Run All.
