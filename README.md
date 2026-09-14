# SMF Batch Job Failure-Risk — Streamlit App

Predicts the failure-risk of mainframe batch jobs from SMF metrics
(EXCP, CPU/queue time, I/O, paging, service class, etc.), based on the
pipeline in `smf_data.ipynb`.

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Rebuilds the RandomForest model from `df_smf.csv` and saves `job_failure_model.pkl` |
| `app.py` | The Streamlit dashboard (loads the pickle, no training at runtime) |
| `requirements.txt` | Python dependencies |
| `job_failure_model.pkl` | Trained model bundle, ready to use |
| `generate_sample_data.py` | Makes a synthetic `df_smf.csv` — **only needed if you don't have the real data file locally** |

## ⚠️ About the included pickle

Your original notebooks load `df_smf.csv` from IBM Cloud Object Storage
using embedded credentials that aren't accessible outside that
notebook environment. Since I couldn't reach your real data here, the
`job_failure_model.pkl` shipped with this app was trained on a
**synthetic dataset** (`generate_sample_data.py`) built to match your
schema, so the app runs end-to-end as a working demo.

**To use your real data:**
```bash
# put your real df_smf.csv in this folder, then:
python train_model.py
```
This regenerates `job_failure_model.pkl` from your actual data —
nothing else needs to change.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (include `job_failure_model.pkl` —
   Git LFS if it grows large).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo.
3. Set the main file to `app.py`. Done — no extra config needed.

## Deploy (any Docker/VM host)

```bash
pip install -r requirements.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## What the model predicts

`FAILURE_RISK` is a proxy label (the notebooks don't have a true
failure flag): the worst 20% of jobs by a weighted rank of elapsed
time, queue time, and EXCP/CPU ratio — computed **within each service
class** — are labelled high-risk. The RandomForest is trained to
reproduce that label from the job's raw SMF metrics, so it flags jobs
that look statistically unusual for their service class before they
finish running.

## Dashboard sections

- **Predict a Job** — manual input form (optionally pre-filled from a
  known JOB_NAME's historical averages) → single risk prediction.
- **Batch Prediction (CSV)** — upload a CSV of jobs, get risk scores
  for all of them plus a downloadable results file.
- **Dashboard** — charts over the training data: risk by service
  class, elapsed-time distribution, jobs by start hour, queue time by
  class, top historically risky jobs.
- **Model Info** — ROC AUC, feature importances, model hyperparameters.
