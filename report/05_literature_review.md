# 4. Literature Review

## 4.1 Recruitment-fraud characterization and data

Vidros *et al.* describe online recruitment fraud, its characteristics, detection methods, and a public dataset [1]. This work is relevant both as background for the threat and as a basis for studying labeled job advertisements. A public benchmark makes reproducible experiments possible, but any detector trained on one historical corpus remains subject to class imbalance, label quality, temporal drift, and domain-specific patterns.

## 4.2 Machine learning and NLP approaches

Amaar *et al.* study fake-job-posting detection using machine-learning and NLP approaches [4]. Their work represents a conventional supervised classification direction in which textual evidence is converted to features for a classifier. Vu *et al.* examine deep-learning-based NLP techniques for fake job-description detection [2], representing a more representation-learning-oriented direction. The two directions have different trade-offs: sparse TF-IDF models can be efficient and inspectable, while deep NLP methods can learn richer contextual representations but require their own data, compute, and validation choices.

## 4.3 Ensemble methods and deployment context

Ullah and Jamjoom propose a secured framework using ensemble machine-learning techniques for online recruitment fraud [3]. Their reported comparison includes AdaBoost, XGBoost, voting, and Random Forest, and the article reports AdaBoost as the strongest of those methods in its experiment. These findings are evidence for the value of comparing multiple algorithms in the authors’ experimental setting; they are not a directly comparable baseline for this project because data handling and evaluation protocols may differ.

Wang *et al.* present a comparative analysis of fraudulent recruitment-advertisement detection methods in an IoT environment [5]. The IoT framing highlights that detection may be considered as part of broader connected systems. This project instead focuses on a local application for a user-entered posting and does not implement an IoT infrastructure or data-collection network.

## 4.4 Research gap and project positioning

The cited studies address recruitment-fraud datasets, conventional ML/NLP, deep NLP, ensemble classifiers, and an IoT context [1]–[5]. A practical gap for this project is the integration of different kinds of evidence in one reproducible workflow: text, structured job metadata, and explicit counts/indicators, followed by validation-only selection, probability calibration, threshold analysis, held-out evaluation, explanation, and local inference. Most importantly, results from different papers cannot establish which approach is superior because this project did not reproduce their datasets and protocols.

The hybrid representation combines five posting-text fields, 11 categorical metadata fields, and 16 deterministic numeric features. It can therefore use information unavailable to a text-only classifier, such as employment metadata, missing-company-profile status, and URL or phone counts. The local text-only LinearSVC baseline and hybrid Linear SVM were evaluated on the same held-out split. The hybrid model had higher fraud recall (0.8256 versus 0.7558), ROC-AUC (0.9856 versus 0.9765), and PR-AUC (0.8958 versus 0.8879), but lower fraud precision (0.8068 versus 0.9353) and F1 (0.8161 versus 0.8360). These results show a useful recall/ranking trade-off, not an across-the-board improvement. This comparison also changes the feature representation, not a fully controlled feature ablation, so it does not isolate the causal contribution of each added feature group.

Among the hybrid candidates, XGBoost achieved test fraud F1 0.8488, compared with 0.8161 for the validation-selected Linear SVM. Selection nevertheless followed the prespecified training-validation PR-AUC criterion (0.9273 for the SVM versus 0.9217 for XGBoost); using test outcomes to switch models would compromise the held-out evaluation. The project’s contribution is therefore a multi-signal, calibrated, threshold-configurable, locally deployable workflow with transparent trade-offs—not a claim of universal superiority over prior work.
