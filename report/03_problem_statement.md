# 2. Problem Statement

## 2.1 Problem

Given the text and available metadata for a job advertisement, estimate whether its recorded target label is fraudulent or legitimate. The project models this as a binary classification task, with `t` mapped to fraud class 1 and `f` mapped to legitimate class 0.

## 2.2 Practical difficulties

- The fraudulent class is rare: after deduplication it represents 858 of 17,645 rows (4.86%).
- Many postings omit optional fields, including department, salary, education, company profile, and benefits.
- Text patterns can overlap across legitimate and fraudulent postings. Remote work, entry-level language, or an incomplete profile is not proof of fraud.
- Fraud indicators have context-dependent meaning and can produce false positives and false negatives.
- The cost of the two error types differs: a false negative may fail to flag a fraudulent posting, while a false positive may wrongly alarm a user about a legitimate opportunity.
- Model output depends on the selected operating threshold and should be interpreted as estimated risk, not a final decision.

## 2.3 Scope

The system analyzes the supplied posting fields locally. It does not crawl the web, verify company registration, inspect a recruiter’s identity, assess the authenticity of linked domains, or determine the intent behind an advertisement. Its output is a model-based estimate derived from labeled historical examples.
