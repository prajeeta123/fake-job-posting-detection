# 11. Future Scope

1. **External and temporal validation:** evaluate on newer data from independent sources and time periods to assess drift and transferability.
2. **Threshold verification:** after choosing the operational threshold using validation data, measure its final performance once on a held-out test set and report those threshold-specific metrics separately.
3. **Model selection robustness:** repeat training-only validation across folds or repeated splits, compare uncertainty intervals, and define the operational objective before examining final test results.
4. **Feature ablation:** compare text-only, metadata-only, engineered-feature-only, and combined models on identical splits to quantify the contribution of each feature group.
5. **Class-imbalance strategy:** evaluate calibrated cost-sensitive objectives or decision policies while keeping resampling and learned transforms confined to training folds.
6. **Fairness and subgroup analysis:** examine errors and calibration across relevant job categories, regions, employment types, and other responsibly selected groups.
7. **Richer contextual language models:** compare modern contextual NLP representations against the existing TF-IDF pipeline with controlled compute and leakage-safe evaluation.
8. **Improve indicators:** review false matches and missed phrases, test context-sensitive patterns, and avoid treating ordinary remote-work or entry-level language as inherently suspicious.
9. **Human feedback and governance:** design privacy-conscious review feedback, model/version documentation, and clear retraining criteria; do not use unverified user reports as labels without review.
10. **Application usability:** improve input guidance and accessibility, explain missing/unknown fields, and provide safe ways to inspect why an alert appeared without implying certainty.
11. **Operational safety:** investigate domain and employer verification as separate, properly sourced checks rather than expanding the classifier's claims beyond its input data.
