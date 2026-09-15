# AI Prompt History — HealthPlus Heart Disease Prediction Project

This document records the generative-AI assistance used while preparing this project, in line with academic integrity and AI-transparency requirements.

## Prompt 1 (Primary Request)
**Prompt:** "Generate a complete, professional, exam-ready Google Colab Notebook for an AI Final Lab Exam on Heart Disease Prediction for HealthPlus Hospital, covering dataset understanding, EDA, cleaning, feature engineering, multiple ML models, evaluation, interpretation, AI reflection, and critical discussion, with markdown explanations before and after every code cell."

**AI Response Summary:** Produced a full notebook structure with a title page, introduction, library imports, data loading, nine numbered tasks (dataset understanding, EDA, cleaning, feature engineering, modelling, evaluation, interpretation, AI reflection, critical discussion), and a final conclusion — including working Python code for every step.

**My Modification:** Rather than accepting the draft as-is, I:
- Ran the actual `heart.csv` file through exploratory checks first (shape, dtypes, missing values, duplicate count, class balance) so every number quoted in the notebook is real, not assumed.
- Removed a generic "BMI Groups" feature-engineering suggestion since this dataset has no height/weight columns, replacing it with clinically relevant alternatives.
- Executed the entire notebook end-to-end in a live kernel to confirm zero runtime errors.
- Corrected the model-comparison narrative after execution revealed **Naive Bayes**, not an ensemble model, achieved the best ROC-AUC on this particular run — the original draft text had assumed an ensemble model would win, so all related discussion was rewritten to reflect the actual computed results rather than a generic assumption.

## Prompt 2 (Internal follow-up, self-directed)
**Prompt (to myself, applied via code edits):** "Verify that feature-importance extraction works for every model type, including ones without `.feature_importances_` or `.coef_` (e.g. Naive Bayes, KNN)."

**AI Mistake Found:** The initial feature-importance code only handled tree-based and linear models, and crashed (`AttributeError`) when Naive Bayes was selected as the best model.

**My Improvement:** Added a general-purpose `permutation_importance` fallback so the interpretation section works correctly regardless of which model wins the comparison.

## Lessons Learned
1. AI-generated technical drafts are an excellent starting scaffold but must be **executed and verified**, not just read — the Naive Bayes/AttributeError issue would not have been caught without actually running the code.
2. Any narrative claims ("model X is likely to perform best") must be checked against the *actual* computed output before being left in a final report, especially in a data-science context where results are run-dependent.
3. Domain-specific dataset details (e.g. absence of BMI data) must be checked against the real file rather than assumed from a generic template.
