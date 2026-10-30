# Real-Time Card Fraud Detection: Geo-Velocity Features, Concept Drift & Cost-Based Thresholds

An end-to-end fraud detection pipeline built on a **synthetic card-transaction simulator**. Public fraud datasets are anonymised (PCA features, no locations or customer IDs), so they can't support the features real fraud teams rely on. This project simulates a card portfolio instead, which makes it possible to engineer **geo-velocity and per-customer behavioural features**, inject a **concept-drift event**, and measure how a model decays and recovers.

Everything lives in one self-contained notebook: [`fraud_detection.ipynb`](fraud_detection.ipynb). Data is generated inside the notebook with a fixed seed, so no downloads are needed.

## Headline results

Evaluated on a held-out, chronologically later test window (final 30 days, 36,423 transactions, 236 frauds), which includes a shift in fraud tactics.

| Finding | Result |
|---|---|
| Best test PR-AUC | Random Forest **0.557** and LightGBM **0.556** (a tie); Logistic Regression 0.410; Isolation Forest 0.217 |
| Lift over random | The random baseline is 0.0065 PR-AUC, so LightGBM is roughly **85x** better |
| Supervised vs. unsupervised | Isolation Forest (no labels) reaches 0.217 PR-AUC, about 2.5x lower than supervised LightGBM |
| Imbalance handling | Class weights, under-sampling and no resampling tie on validation (about 0.93 PR-AUC); SMOTE was slightly worse (0.910) |
| Cost reduction vs. no model | **57.7%** at the cost-optimal threshold ($22,214 of fraud losses down to $9,398) |
| Cost reduction at F1-optimal threshold | Only **42.2%**, because F1 treats a $5 false alarm the same as a missed fraud |
| Concept drift | PR-AUC falls from **0.94 to 0.36-0.47** once fraudsters change tactics; weekly retraining recovers part of it (up to **+0.11** PR-AUC in the final week) |
| Explainability | Per-alert SHAP reason codes in plain English, such as "implied travel speed of 1,571 km/h" and "3,501 km from the customer's usual location" |

## The simulated data

108,310 transactions from 1,500 customers over 90 days (2025-01-01 to 2025-03-31), with 710 frauds (0.66%).

**Legitimate behaviour:** each customer has a home city, typical spend level, favourite merchant categories and a daily rhythm. About 50% take legitimate trips to other cities, about 10% are night owls, and about 4% of purchases are unusually large. These create "far from home", "2 am" and "big amount" transactions that are *not* fraud, which is what makes the problem hard.

**Fraud patterns:**

| Pattern | Description | Count |
|---|---|---|
| A: Geo-hop | Purchase in a far-away city 1-6 hours after a legitimate local purchase ("impossible travel") | 164 |
| B: Night spree | Several larger-than-usual purchases between 00:00 and 05:00 | 207 |
| C: Card testing | Burst of tiny online purchases within minutes | 207 |
| D: Low-and-slow | Moderate, daytime, *local* purchases; rare before day 70, dominant after it, to simulate **concept drift** | 132 |

The `fraud_type` label is used for analysis only and is never a model feature.

## Approach

1. **Simulate** transactions with realistic hard negatives and a drift event.
2. **EDA.** Fraud *rate* peaks in the low-volume overnight hours (up to about 4.5% vs. 0.66% overall), which raw fraud counts hide.
3. **Leakage-free feature engineering** (22 features), computed only from each customer's *past* transactions:
   - Geo-velocity: haversine distance and implied travel speed from the previous transaction
   - Distance from the customer's usual location (expanding median of past locations)
   - Amount relative to the customer's recent average
   - Velocity counts (transactions in the last hour and 24 hours)
   - First purchase in a merchant category, hour, day of week, night flag, category dummies
4. **Chronological split.** Train on days 0-44 (53,712 transactions, 350 frauds), validate on days 45-59 (18,175; 124 frauds), test on days 60-89 (36,423; 236 frauds).
5. **Imbalance strategies.** None, class weights, under-sampling and SMOTE, compared on validation PR-AUC with the same LightGBM model.
6. **Model comparison.** Logistic Regression, Random Forest, LightGBM and Isolation Forest, evaluated with PR-AUC, ROC-AUC and recall at 80% precision.
7. **Cost-based threshold.** A missed fraud costs its transaction amount and a false alarm costs a flat $5. The threshold is tuned on validation and applied unchanged to test.
8. **Concept drift.** A static model (trained once) is compared with a model retrained weekly on all history.
9. **SHAP.** Global feature importance, per-alert reason codes and a waterfall plot.

## Key findings

- **Accuracy is meaningless here.** Predicting "legitimate" for everything scores over 99% accuracy while catching no fraud, so models are judged on PR-AUC, recall at fixed precision, and dollars.
- **Choosing the metric that matches the cost matters.** The F1-optimal threshold (0.839) caught 110 of 236 frauds and cost $12,834. The cost-optimal threshold (0.067) caught 129 and cost $9,398, saving about $3,400 more on the test window. The default 0.5 threshold scored close to the cost-optimal one (55.8% vs. 57.7% savings).
- **Models decay under drift.** When pattern D appears (53-65% of frauds in the final three weeks), the static model's PR-AUC drops from 0.936 to 0.36-0.47. Retraining helps, but the gain is small in the first drift week (+0.01), +0.04 in the second and +0.11 in the final week, and it never fully closes the gap.
- **The engineered features drive the alerts.** In the per-alert SHAP explanations, implied travel speed, distance from the customer's usual location and time since the previous transaction lead the reason codes, with transaction amount and category also ranking high globally.
- **Burst activity is caught by velocity features.** One flagged fraud was explained mainly by five other transactions in the past hour, even though its implied travel speed was only 38 km/h.

## Limitations

- **Synthetic data.** The fraud patterns are ones I defined, so absolute scores overstate real-world performance and the SHAP findings reflect the simulator's rules. What transfers is the methodology: leakage-free features, time-based validation, cost-based thresholds and drift monitoring.
- **Label delay.** Real fraud labels (chargebacks) arrive days or weeks late. The retraining experiment assumes instant labels.
- **Simplified costs.** A flat $5 review cost and "loss = amount" ignore chargeback fees, recovery rates and customer lifetime value.
- **Noisy comparisons.** Weekly drift results rest on 53-72 frauds per week, and no confidence intervals are reported.
- **Threshold stability.** The threshold was tuned on pre-drift validation data and would need re-tuning under drift.
- **Batch only.** Feature computation is batch; a deployed system needs a feature store and sub-100 ms scoring.

## Possible extensions

- Run the same pipeline on a real dataset with richer fields (Sparkov, IEEE-CIS).
- Add bootstrap confidence intervals for model and drift comparisons.
- Train retraining models only on labels available at retraining time (label delay).
- Probability calibration and an amount-aware alert rule.
- Graph features to detect shared-device or shared-merchant fraud rings.
- Serve the model behind a FastAPI endpoint.

## Run it

```bash
git clone <your-repo-url>
cd <your-repo>
pip install numpy pandas scikit-learn lightgbm shap imbalanced-learn matplotlib seaborn jupyter
jupyter notebook fraud_detection.ipynb
```

Seeds are fixed (seed 42), so results should be reproducible apart from small differences across library versions. The full run takes a few minutes.

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, imbalanced-learn (SMOTE), SHAP, matplotlib, seaborn.
