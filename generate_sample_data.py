"""
Generates a synthetic df_smf.csv that matches the schema of the real
mainframe SMF batch-job dataset used in the original notebooks
(predition_algorithm__1_.ipynb / smf_data.ipynb).

Why this exists
----------------
The original notebooks pull `df_smf.csv` from IBM Cloud Object Storage
using embedded API credentials. Those credentials/bucket aren't
reachable from here, so this script fabricates a realistic
stand-in dataset with the same columns and roughly the same
statistical shape, purely so that `train_model.py` has something to
train on and the Streamlit app can be demoed end-to-end.

>>> REPLACE THIS WITH YOUR REAL df_smf.csv BEFORE GOING TO PRODUCTION. <<<
Just drop your real file in this folder as `df_smf.csv` and re-run
`train_model.py` - nothing else needs to change.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_ROWS = 4000

rng = np.random.default_rng(RANDOM_STATE)

SERV_CLASSES = ["BATCH_A", "BATCH_B", "BATCH_C", "ONLINE_A", "STC_A"]
CLASS_VALUES = [100, 200, 300, 400]

n_jobs = 350
job_names = [f"JN_{i:03d}" for i in range(1, n_jobs + 1)]

rows = []
start = pd.Timestamp("2018-07-31 00:00:00")

for i in range(N_ROWS):
    job_name = rng.choice(job_names)
    serv_class = rng.choice(SERV_CLASSES, p=[0.35, 0.25, 0.15, 0.15, 0.10])
    cls = int(rng.choice(CLASS_VALUES))

    # base workload intensity varies a bit by service class
    intensity = {"BATCH_A": 1.0, "BATCH_B": 1.6, "BATCH_C": 2.4,
                 "ONLINE_A": 0.5, "STC_A": 0.3}[serv_class]

    excp = int(max(0, rng.normal(80_000, 60_000) * intensity))
    io_conn_sec = int(max(0, rng.normal(120, 90) * intensity))
    page_in = int(max(0, rng.exponential(50)))
    page_out = int(max(0, rng.exponential(20)))
    page_swap = int(max(0, rng.exponential(2)))
    ssch = int(max(0, rng.normal(3000, 2500) * intensity))
    tcb_cpu_sec = round(max(0, rng.normal(60, 50) * intensity), 2)
    srb_cpu_sec = round(max(0, rng.normal(3, 4) * intensity), 2)
    cpu_sec = round(tcb_cpu_sec + srb_cpu_sec, 2)
    serv_unit = int(max(0, rng.normal(150_000, 120_000) * intensity))
    mso_unit = int(max(0, rng.normal(8_000, 6_000) * intensity))
    jq_sec = int(max(0, rng.exponential(5)))
    hq_sec = int(max(0, rng.exponential(3)))

    minute_offset = int(rng.integers(0, 60 * 24 * 5))  # spread over 5 days
    start_ts = start + pd.Timedelta(minutes=minute_offset,
                                     seconds=int(rng.integers(0, 60)))
    elapsed_sec = round(max(0.1, rng.normal(180, 220) * intensity), 2)
    end_ts = start_ts + pd.Timedelta(seconds=elapsed_sec)

    rows.append({
        "CLASS": cls,
        "SERV_CLASS": serv_class,
        "JOB_NAME": job_name,
        "JOB_NUM": f"JOB{rng.integers(30000, 40000)}",
        "START_DTSTR": start_ts.strftime("%Y-%m-%d-%H.%M.%S.%f"),
        "END_DTSTR": end_ts.strftime("%Y-%m-%d-%H.%M.%S.%f"),
        "ELAPSED_SEC": elapsed_sec,
        "EXCP": excp,
        "IO_CONN_SEC": io_conn_sec,
        "PAGE_IN": page_in,
        "PAGE_OUT": page_out,
        "PAGE_SWAP": page_swap,
        "SSCH": ssch,
        "TCB_CPU_SEC": tcb_cpu_sec,
        "SRB_CPU_SEC": srb_cpu_sec,
        "CPU_SEC": cpu_sec,
        "SERV_UNIT": serv_unit,
        "MSO_UNIT": mso_unit,
        "JQ_SEC": jq_sec,
        "HQ_SEC": hq_sec,
    })

df = pd.DataFrame(rows)
df.to_csv("df_smf.csv", index=False)
print(f"Wrote df_smf.csv with {len(df)} rows and {len(df.columns)} columns")
print(df.head())
