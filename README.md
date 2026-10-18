# Estimating the Likelihood of a Reported COPD Diagnosis in Adults 50+

Traditional machine learning versus deep learning on the 2023 CDC Behavioral Risk Factor
Surveillance System (BRFSS) survey.

This project estimates whether an adult aged 50 or older reports a physician diagnosis of COPD
(chronic obstructive pulmonary disease, emphysema or chronic bronchitis), using only 38
self-reported survey answers (smoking history, general health, comorbid conditions, healthcare
use and background). It compares four scikit-learn models with a family of Keras neural networks
under one shared data split, and looks at who the models fail to detect.

> **Not a medical device.** The target is a *self-reported diagnosis*, not spirometry. The models
> are meant as a research example of a screening aid for prioritising confirmatory testing, not
> for diagnosing anyone.

## Project links

| Item | Link |
|---|---|
| Presentation video | [https://drive.google.com/drive/folders/1RB6GR0xffOuHwQV2O7Lhy1UMDmKeRnsv?usp=sharing] |
| Google Drive folder (data, models, results) | [https://drive.google.com/drive/folders/1C1CZky0LJRitftbmIahU-fZLnZ1BjHtz?usp=sharing] |
| Experiment tracking (Weights & Biases, optional) | [https://forge.coreweave.com/wandb/k-kamikazi-african-leadership-university/copd-2023] |

## Key results

Held-out test set (15% of respondents, 40,388 people, 10.56% positive):

| Model | ROC-AUC | PR-AUC | Recall | Precision | People flagged |
|---|---|---|---|---|---|
| Logistic regression (tuned) | 0.853 | 0.464 | 0.756 | 0.304 | 26.3% |
| Random forest (tuned) | 0.854 | 0.465 | 0.781 | 0.288 | 28.7% |
| **Histogram gradient boosting (tuned)** | **0.859** | **0.480** | 0.767 | 0.312 | 26.0% |
| SVM (RBF, 15,000-row subsample) | 0.823 | 0.374 | 0.765 | 0.254 | 31.9% |
| Neural network (embeddings, 128-64) | 0.857 | 0.480 | 0.768 | 0.302 | 26.9% |

Operating thresholds are chosen on the validation set by maximising F2 (recall weighted twice as
heavily as precision) and then applied unchanged to the test set.

Main findings:

- Tuned gradient boosting and the best neural network tie on PR-AUC (0.480 each). A paired
  bootstrap (200 resamples) finds no difference in PR-AUC.
- Network size, learning rate, batch size and embeddings changed validation PR-AUC by less than
  0.005. Removing regularisation was the only change that clearly hurt (PR-AUC 0.476).
- The models find about 77% of reported cases while flagging about 26% of people, with 31%
  precision (roughly three times the 10.6% base rate).
- Recall is much lower for never-smokers (0.48) and for respondents with incomes above $75,000
  (0.55). Never-smokers make up about half of all missed cases.

## Repository contents

```
.
├── README.md
├── requirements.txt
├── copd_notebook.ipynb                              
```

Large files (the raw BRFSS file, the cleaned Parquet file and the saved models) are **not**
stored in the repository. They live in the Google Drive folder linked above.

## Data

- **Source:** CDC, *2023 BRFSS Survey Data and Documentation*
  (<https://www.cdc.gov/brfss/annual_data/annual_2023.html>). The notebook uses the SAS Transport
  (XPT) file, `LLCP2023XPT.zip`, which contains 433,323 respondent records and about 350 columns.
- **Licence and privacy:** the data are a public U.S. government survey release. No personal
  identifiers are used, and no private data were collected.
- **Cohort used here:** respondents with a valid yes/no COPD answer (431,257), restricted to
  age 50 and over, giving 269,252 respondents of whom 28,432 (10.56%) report a COPD diagnosis.
- **Not included:** race and one conditionally asked health-days question were deliberately left
  out of the predictors (see the report, Section 3.1).

## How to run

### Option 1: Google Colab (the setup used for the reported results)

1. Download `LLCP2023XPT.zip` from the CDC page above.
2. Put it in Google Drive at `MyDrive/ml_project/LLCP2023XPT.zip`.
3. Open `notebooks/copd_notebook.ipynb` in Colab and choose **Runtime > Run all**.
   - Part 1 reads the XPT file in chunks, builds the cohort and saves
     `brfss_copd_50plus.parquet` to Drive.
   - Part 2 trains and evaluates the traditional models and saves the best one.
   - Part 3 trains and evaluates the neural networks and compares them with Part 2.
4. For a fast smoke test, set `QUICK_RUN = True` in the setup cell. This uses 12% of the data and
   fewer search draws, and writes files with a `_quick` suffix so the full results are not
   overwritten.

The notebook ran on CPU only (TensorFlow reported no GPU). The longest steps in the reported run
were the random-forest search (about 8 minutes) and the boosting search (about 6 minutes), and
the individual neural-network runs took roughly 2 to 6 minutes each.

### Option 2: Local machine

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

The notebook mounts Google Drive and uses `/content/...` paths. To run locally, skip the
`drive.mount` cell and change `RAW_PATH`, `OUT_PATH` and `MODEL_DIR` in the setup cell to local
folders. Unzip `LLCP2023XPT.zip` first; the file inside is named `LLCP2023.XPT` (it may have a
trailing space in its name, which the notebook handles).

### Weights & Biases (optional)

Experiment tracking is on by default (`USE_WANDB = True`, project `copd-2023`). Choose one:

- Add a secret named `WANDB_API_KEY` (Colab Secrets or Kaggle Secrets), or
- run offline with `export WANDB_MODE=offline`, or
- set `USE_WANDB = False` in the setup cell.

## Notebook structure

| Part | Content |
|---|---|
| 0. Setup | Imports, configuration (`SEED = 42`, paths, `QUICK_RUN`), W&B |
| 1. Data | Variable list, chunked loading, cohort definition, special-value recoding, missing-value profile, save cleaned cohort |
| 2. Traditional ML | Feature groups, stratified 70/15/15 split, leakage-safe pipelines, four candidate models, missing-value and class-weight experiments, randomised-search tuning, learning curve, test evaluation, threshold analysis, permutation importance and ablation, subgroup error analysis, saving the best model |
| 3. Deep learning | `tf.data` pipelines, Keras Functional API models, experiments on architecture, regularisation, optimisation and embeddings, learning curves, test comparison with a paired bootstrap, threshold and subgroup analysis, saving the model |

## Files written by the notebook

In `MODEL_DIR` (default `MyDrive/ml_project/models`):

- `best_traditional_model.joblib`, `best_traditional_meta.json`
- `split_indices.npz` (train/validation/test row indices, so both parts use identical rows)
- `traditional_test_results.csv`, `dl_experiment_results.csv`
- `best_dl_model.keras` (plus `.h5` and a SavedModel export when they succeed)
- `dl_preprocessing.joblib`, `best_dl_meta.json`

Summary tables (tuning, ablation, threshold analysis, bootstrap differences and others) are saved
as CSV files in a sibling `results/` folder.

## Reproducibility

- A fixed seed (`SEED = 42`) controls the split, the searches and the network initialisation.
- The split is saved to `split_indices.npz`, and Part 3 asserts that it matches the one from
  Part 2.
- All preprocessing is fitted on training rows only, inside pipelines.
- Each neural-network configuration was trained once, so run-to-run variation (informally about
  0.004 PR-AUC across configurations) was not measured directly. Small differences between
  configurations should not be over-interpreted.
- Exact package versions: only TensorFlow 2.21.0 and Python 3.12.3 were recorded. To freeze the
  versions from your own run, execute this in a notebook cell and paste the output into
  `requirements.txt`:

```python
import numpy, pandas, scipy, sklearn, matplotlib, joblib, pyarrow, tensorflow, wandb
for m in (numpy, pandas, scipy, sklearn, matplotlib, joblib, pyarrow, tensorflow, wandb):
    print(f"{m.__name__}=={m.__version__}")
```

## Limitations

- The outcome is a self-reported diagnosis, so undiagnosed cases are labelled negative and the
  label reflects access to care as well as disease.
- BRFSS is a cross-sectional telephone survey, so results are associations at one point in time.
- The SVM used a 15,000-row subsample and tuning used a 60,000-row subsample.
- Probability calibration and external validation on other survey years were not assessed.

## Acknowledgments

- Data: CDC BRFSS 2023 (public).
- Libraries: scikit-learn, TensorFlow/Keras, pandas, NumPy, SciPy, Matplotlib, joblib, PyArrow,
  Weights & Biases. No pretrained models were used.

## Author

KELLIA KAMIKAZI (k.kamikazi@alustudent.com)
