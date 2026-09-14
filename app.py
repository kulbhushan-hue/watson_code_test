"""
Streamlit dashboard for the SMF Batch Job Failure-Risk model.

Run locally:
    streamlit run app.py

Expects `job_failure_model.pkl` (built by train_model.py) in the
same folder as this file.
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------
st.set_page_config(
    page_title="SMF Batch Job Failure Risk",
    page_icon="🖥️",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(128,128,128,0.08);
        border-radius: 10px;
        padding: 10px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact(path: str = "job_failure_model.pkl"):
    return joblib.load(path)


try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error(
        "Couldn't find `job_failure_model.pkl` in the app folder. "
        "Run `python train_model.py` first to create it."
    )
    st.stop()

model = artifact["model"]
le_servclass = artifact["label_encoder_servclass"]
FEATURE_COLS = artifact["feature_cols"]
NUMERIC_FEATURES = artifact["numeric_features"]
job_profile = artifact["job_profile"]
global_defaults = artifact["global_defaults"]
serv_class_options = artifact["serv_class_options"]
class_options = artifact["class_options"]
job_name_options = artifact["job_name_options"]
metrics = artifact["training_metrics"]
raw_df = artifact["raw_df"]


def encode_serv_class(value: str) -> int:
    """LabelEncoder-safe transform; falls back to 0 for unseen classes."""
    if value in le_servclass.classes_:
        return int(le_servclass.transform([value])[0])
    return 0


def make_feature_row(values: dict) -> pd.DataFrame:
    row = {c: values.get(c, 0) for c in FEATURE_COLS}
    return pd.DataFrame([row], columns=FEATURE_COLS)


def predict_risk(values: dict):
    X = make_feature_row(values)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    return pred, proba


def risk_color(proba: float) -> str:
    if proba >= 0.66:
        return "#e03131"   # red
    if proba >= 0.33:
        return "#f2994a"   # amber
    return "#2f9e44"       # green


def risk_gauge(proba: float, title: str = "Predicted Failure-Risk", height: int = 300, key_suffix: str = ""):
    """Circular gauge (donut-style dial) showing risk probability 0-100%."""
    color = risk_color(proba)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(47,158,68,0.18)"},
                    {"range": [33, 66], "color": "rgba(242,153,74,0.18)"},
                    {"range": [66, 100], "color": "rgba(224,49,49,0.18)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.9,
                    "value": proba * 100,
                },
            },
        )
    )
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{key_suffix}")


def risk_donut(n_low, n_med, n_high, key_suffix: str = ""):
    """Donut chart breaking a batch of jobs into risk tiers."""
    fig = go.Figure(
        go.Pie(
            labels=["Low", "Medium", "High"],
            values=[n_low, n_med, n_high],
            hole=0.6,
            marker=dict(colors=["#2f9e44", "#f2994a", "#e03131"]),
            sort=False,
        )
    )
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True, legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"donut_{key_suffix}")


# --------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------
st.sidebar.title("🖥️ SMF Job Monitor")
page = st.sidebar.radio(
    "Go to",
    ["Predict a Job", "Batch Prediction (CSV)", "Dashboard", "Model Info"],
)

st.sidebar.markdown("---")
st.sidebar.metric("Model ROC AUC", f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] else "n/a")
st.sidebar.metric("Historical failure-risk rate", f"{metrics['label_positive_rate']*100:.1f}%")
st.sidebar.caption(f"Trained on {metrics['n_train']} jobs · tested on {metrics['n_test']} jobs")

# --------------------------------------------------------------------
# PAGE 1 — Predict a single job
# --------------------------------------------------------------------
if page == "Predict a Job":
    st.title("Predict Failure Risk for a Batch Job")
    st.caption(
        "Enter job metrics manually, or pick a known JOB_NAME to pre-fill "
        "its historical averages, then adjust as needed."
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        job_choice = st.selectbox(
            "Pre-fill from a known job (optional)",
            ["— none, start blank —"] + job_name_options,
        )

    if job_choice != "— none, start blank —":
        profile_row = job_profile.loc[job_profile["JOB_NAME"] == job_choice].iloc[0]
        defaults = profile_row.to_dict()
        default_serv_class = defaults.get("SERV_CLASS", serv_class_options[0])
        st.info(
            f"**{job_choice}** has run {int(profile_row['RUN_COUNT'])} times historically, "
            f"with an average failure-risk rate of {profile_row['FAILURE_RISK']*100:.0f}%."
        )
    else:
        defaults = global_defaults
        default_serv_class = serv_class_options[0]

    st.subheader("Job parameters")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            serv_class = st.selectbox(
                "Service class (SERV_CLASS)",
                serv_class_options,
                index=serv_class_options.index(default_serv_class)
                if default_serv_class in serv_class_options else 0,
            )
            job_class = st.selectbox(
                "CLASS",
                class_options,
                index=class_options.index(defaults.get("CLASS", class_options[0]))
                if defaults.get("CLASS", class_options[0]) in class_options else 0,
            )
            start_hour = st.slider("Start hour of day", 0, 23, int(defaults.get("START_HOUR", 12)) if "START_HOUR" in defaults else 12)
            start_dow = st.slider("Start day of week (0=Mon)", 0, 6, int(defaults.get("START_DOW", 2)) if "START_DOW" in defaults else 2)

        with c2:
            excp = st.number_input("EXCP", min_value=0.0, value=float(defaults.get("EXCP", global_defaults["EXCP"])))
            io_conn_sec = st.number_input("IO_CONN_SEC", min_value=0.0, value=float(defaults.get("IO_CONN_SEC", global_defaults["IO_CONN_SEC"])))
            ssch = st.number_input("SSCH", min_value=0.0, value=float(defaults.get("SSCH", global_defaults["SSCH"])))
            serv_unit = st.number_input("SERV_UNIT", min_value=0.0, value=float(defaults.get("SERV_UNIT", global_defaults["SERV_UNIT"])))
            mso_unit = st.number_input("MSO_UNIT", min_value=0.0, value=float(defaults.get("MSO_UNIT", global_defaults["MSO_UNIT"])))

        with c3:
            tcb_cpu_sec = st.number_input("TCB_CPU_SEC", min_value=0.0, value=float(defaults.get("TCB_CPU_SEC", global_defaults["TCB_CPU_SEC"])))
            srb_cpu_sec = st.number_input("SRB_CPU_SEC", min_value=0.0, value=float(defaults.get("SRB_CPU_SEC", global_defaults["SRB_CPU_SEC"])))
            cpu_sec = st.number_input("CPU_SEC", min_value=0.0, value=float(defaults.get("CPU_SEC", global_defaults["CPU_SEC"])))
            page_in = st.number_input("PAGE_IN", min_value=0.0, value=float(defaults.get("PAGE_IN", global_defaults["PAGE_IN"])))
            page_out = st.number_input("PAGE_OUT", min_value=0.0, value=float(defaults.get("PAGE_OUT", global_defaults["PAGE_OUT"])))
            page_swap = st.number_input("PAGE_SWAP", min_value=0.0, value=float(defaults.get("PAGE_SWAP", global_defaults["PAGE_SWAP"])))
            total_queue_sec = st.number_input("TOTAL_QUEUE_SEC (JQ_SEC + HQ_SEC)", min_value=0.0, value=float(defaults.get("TOTAL_QUEUE_SEC", global_defaults["TOTAL_QUEUE_SEC"])))

        submitted = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

    if submitted:
        values = {
            "EXCP": excp, "IO_CONN_SEC": io_conn_sec, "PAGE_IN": page_in,
            "PAGE_OUT": page_out, "PAGE_SWAP": page_swap, "SSCH": ssch,
            "TCB_CPU_SEC": tcb_cpu_sec, "SRB_CPU_SEC": srb_cpu_sec,
            "SERV_UNIT": serv_unit, "MSO_UNIT": mso_unit,
            "TOTAL_QUEUE_SEC": total_queue_sec, "CPU_SEC": cpu_sec,
            "START_HOUR": start_hour, "START_DOW": start_dow, "CLASS": job_class,
            "SERV_CLASS_ENC": encode_serv_class(serv_class),
        }
        pred, proba = predict_risk(values)
        st.session_state["last_prediction"] = {"values": values, "pred": pred, "proba": proba}

    if "last_prediction" in st.session_state:
        pred = st.session_state["last_prediction"]["pred"]
        proba = st.session_state["last_prediction"]["proba"]
        values = st.session_state["last_prediction"]["values"]

        st.markdown("---")
        g1, g2 = st.columns([1, 2])
        with g1:
            risk_gauge(proba, key_suffix="single")
            label = "HIGH RISK" if proba >= 0.66 else ("MEDIUM RISK" if proba >= 0.33 else "LOW RISK")
            st.markdown(
                f"<h4 style='text-align:center; color:{risk_color(proba)};'>{label}</h4>",
                unsafe_allow_html=True,
            )
        with g2:
            st.write("**How this job compares to the training feature importances:**")
            importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(8)
            imp_df = importances.reset_index()
            imp_df.columns = ["feature", "importance"]
            fig = px.bar(
                imp_df, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Oranges",
            )
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True, key="importance_single")

        with st.expander("Show input values sent to the model"):
            st.json(values)

# --------------------------------------------------------------------
# PAGE 2 — Batch prediction from an uploaded CSV
# --------------------------------------------------------------------
elif page == "Batch Prediction (CSV)":
    st.title("Batch Prediction from CSV")
    st.caption(
        "Upload a CSV of jobs (same raw columns as df_smf.csv — including "
        "SERV_CLASS, START_DTSTR, JQ_SEC, HQ_SEC, etc.) to score them all at once."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)

            if "START_DTSTR" in new_df.columns:
                new_df["START_DTSTR"] = pd.to_datetime(
                    new_df["START_DTSTR"], format="%Y-%m-%d-%H.%M.%S.%f", errors="coerce"
                )
                new_df["START_HOUR"] = new_df["START_DTSTR"].dt.hour
                new_df["START_DOW"] = new_df["START_DTSTR"].dt.dayofweek

            if {"JQ_SEC", "HQ_SEC"}.issubset(new_df.columns):
                new_df["TOTAL_QUEUE_SEC"] = new_df["JQ_SEC"] + new_df["HQ_SEC"]

            if "SERV_CLASS" in new_df.columns:
                new_df["SERV_CLASS_ENC"] = new_df["SERV_CLASS"].apply(encode_serv_class)

            missing = [c for c in FEATURE_COLS if c not in new_df.columns]
            for c in missing:
                new_df[c] = global_defaults.get(c, 0)
            if missing:
                st.warning(f"Missing columns filled with training-set medians: {missing}")

            X_new = new_df[FEATURE_COLS].fillna(0)
            new_df["FAILURE_RISK_PRED"] = model.predict(X_new)
            new_df["FAILURE_RISK_PROBA"] = model.predict_proba(X_new)[:, 1]
            new_df["RISK_TIER"] = pd.cut(
                new_df["FAILURE_RISK_PROBA"], bins=[-0.01, 0.33, 0.66, 1.01],
                labels=["Low", "Medium", "High"],
            )

            st.success(f"Scored {len(new_df)} jobs.")

            n_low = int((new_df["RISK_TIER"] == "Low").sum())
            n_med = int((new_df["RISK_TIER"] == "Medium").sum())
            n_high = int((new_df["RISK_TIER"] == "High").sum())
            avg_proba = float(new_df["FAILURE_RISK_PROBA"].mean())

            g1, g2, g3 = st.columns([1, 1, 2])
            with g1:
                risk_gauge(avg_proba, title="Batch Avg. Risk", height=260, key_suffix="batch")
            with g2:
                risk_donut(n_low, n_med, n_high, key_suffix="batch")
            with g3:
                st.metric("Jobs scored", len(new_df))
                st.metric("Flagged high-risk", n_high, delta=f"{n_high/len(new_df)*100:.1f}% of batch")
                st.metric("Flagged medium-risk", n_med)

            st.markdown("---")
            st.subheader("Risk probability distribution")
            fig = px.histogram(
                new_df, x="FAILURE_RISK_PROBA", nbins=20, color="RISK_TIER",
                color_discrete_map={"Low": "#2f9e44", "Medium": "#f2994a", "High": "#e03131"},
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), bargap=0.05)
            st.plotly_chart(fig, use_container_width=True, key="batch_hist")

            st.subheader("Scored jobs")
            tier_filter = st.multiselect("Filter by risk tier", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
            show_cols = [c for c in ["JOB_NAME", "SERV_CLASS", "CLASS"] if c in new_df.columns]
            show_cols += ["FAILURE_RISK_PRED", "FAILURE_RISK_PROBA", "RISK_TIER"]
            filtered = new_df[new_df["RISK_TIER"].isin(tier_filter)]
            st.dataframe(
                filtered[show_cols].sort_values("FAILURE_RISK_PROBA", ascending=False),
                use_container_width=True,
            )

            st.download_button(
                "Download scored CSV",
                new_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_jobs.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Couldn't process this file: {e}")
    else:
        st.info("Waiting for a CSV upload.")

# --------------------------------------------------------------------
# PAGE 3 — Dashboard / analytics over the training data
# --------------------------------------------------------------------
elif page == "Dashboard":
    st.title("Batch Job Analytics Dashboard")

    with st.expander("Filters", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            class_filter = st.multiselect(
                "Service class", options=sorted(raw_df["SERV_CLASS"].unique().tolist()),
                default=sorted(raw_df["SERV_CLASS"].unique().tolist()),
            )
        with f2:
            hour_range = st.slider("Start hour range", 0, 23, (0, 23))

    fdf = raw_df[
        raw_df["SERV_CLASS"].isin(class_filter)
        & raw_df["START_HOUR"].between(hour_range[0], hour_range[1])
    ]

    n_low = int((fdf["RISK_SCORE"] < fdf["RISK_SCORE"].quantile(0.66)).sum()) if len(fdf) else 0
    n_high = int((fdf["FAILURE_RISK"] == 1).sum()) if len(fdf) else 0
    n_total = len(fdf)
    n_med = max(n_total - n_low - n_high, 0)
    avg_risk = float(fdf["FAILURE_RISK"].mean()) if n_total else 0.0

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        risk_gauge(avg_risk, title="Overall Risk Level", height=260, key_suffix="dash")
    with c2:
        risk_donut(n_low, n_med, n_high, key_suffix="dash")
    with c3:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jobs (filtered)", n_total)
        m2.metric("Unique job names", fdf["JOB_NAME"].nunique() if n_total else 0)
        m3.metric("Service classes", fdf["SERV_CLASS"].nunique() if n_total else 0)
        m4.metric("High-risk jobs", n_high)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Failure risk by service class")
        risk_by_class = fdf.groupby("SERV_CLASS")["FAILURE_RISK"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(risk_by_class, x="SERV_CLASS", y="FAILURE_RISK", color="FAILURE_RISK", color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, key="dash_bar1")

    with right:
        st.subheader("Elapsed time distribution")
        fig = px.histogram(
            fdf[fdf["ELAPSED_SEC"] <= fdf["ELAPSED_SEC"].quantile(0.95)],
            x="ELAPSED_SEC", nbins=25, color_discrete_sequence=["#4c6ef5"],
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), bargap=0.05)
        st.plotly_chart(fig, use_container_width=True, key="dash_hist1")

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Jobs by start hour")
        by_hour = fdf["START_HOUR"].value_counts().sort_index().reset_index()
        by_hour.columns = ["START_HOUR", "count"]
        fig = px.line(by_hour, x="START_HOUR", y="count", markers=True)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, key="dash_line1")
    with right2:
        st.subheader("Average queue time by service class")
        by_q = fdf.groupby("SERV_CLASS")["TOTAL_QUEUE_SEC"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(by_q, x="SERV_CLASS", y="TOTAL_QUEUE_SEC", color="TOTAL_QUEUE_SEC", color_continuous_scale="Blues")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, key="dash_bar2")

    st.markdown("---")
    st.subheader("Highest-risk jobs on record")
    top_risky = job_profile.sort_values("FAILURE_RISK", ascending=False).head(10)
    st.dataframe(
        top_risky[["JOB_NAME", "SERV_CLASS", "RUN_COUNT", "FAILURE_RISK", "ELAPSED_SEC"]],
        use_container_width=True,
    )

# --------------------------------------------------------------------
# PAGE 4 — Model info
# --------------------------------------------------------------------
elif page == "Model Info":
    st.title("Model Information")

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC AUC", f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] else "n/a")
    c2.metric("Training rows", metrics["n_train"])
    c3.metric("Test rows", metrics["n_test"])

    st.subheader("Feature importance")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    imp_df = importances.reset_index()
    imp_df.columns = ["feature", "importance"]
    fig = px.bar(imp_df, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Oranges")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True, key="model_info_imp")

    st.dataframe(imp_df)

    st.subheader("Model configuration")
    st.json({k: v for k, v in model.get_params().items()})

    st.caption(
        "FAILURE_RISK is a proxy label: the top 20% of jobs by a weighted "
        "rank of elapsed time, queue time, and EXCP/CPU ratio (computed "
        "within each service class) are labelled high-risk."
    )
