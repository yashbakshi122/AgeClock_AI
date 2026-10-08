import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AgeClock AI",
    page_icon="🧬",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0b1020;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #111827, #172554);
    border: 1px solid #263b70;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
    color: #cbd5e1;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    background: #111827;
    border: 1px solid #334155;
    text-align: center;
}

.result-number {
    font-size: 42px;
    font-weight: bold;
}

.small-text {
    color: #94a3b8;
}

.warning {
    padding: 15px;
    border-radius: 12px;
    background: #3f3a12;
    border: 1px solid #756b18;
    color: #fef08a;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "age_clock_model.joblib"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "model",
    "feature_names.json"
)

INFO_PATH = os.path.join(
    BASE_DIR,
    "model",
    "age_clock_model_info.json"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    with open(FEATURE_PATH, "r") as f:
        feature_names = json.load(f)

    return model, feature_names


@st.cache_data
def load_model_info():

    if os.path.exists(INFO_PATH):

        with open(INFO_PATH, "r") as f:
            return json.load(f)

    return {}


model, feature_names = load_model()
model_info = load_model_info()


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>🧬 AgeClock AI</h1>

<p>
Gene-expression based molecular age prediction
</p>

<p>
Upload a GEO-style gene-expression file and estimate
molecular / chronological age using a machine-learning model.
</p>

</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="warning">

⚠️ <b>Research / Educational Project</b><br>
This tool is a research prototype and is <b>not a medical diagnostic system</b>.
Predictions should not be used for medical decisions.

</div>
""", unsafe_allow_html=True)


st.write("")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🧬 AgeClock AI")

    st.markdown("""
    **Pipeline**

    Gene Expression  
    ↓  
    Feature Selection  
    ↓  
    Standardization  
    ↓  
    Ridge Regression  
    ↓  
    Predicted Age
    """)

    st.divider()

    st.subheader("Model")

    st.write("Selected features: 100")

    st.write("Cross-validation: GroupKFold")

    st.write("Twin pairs: 27")

    st.divider()

    st.caption(
        "Built using GSE74937 gene-expression data."
    )


# =========================================================
# FILE UPLOAD
# =========================================================

st.subheader("📁 Upload Gene-Expression Data")

uploaded_file = st.file_uploader(
    "Upload a GEO-style gene-expression file",
    type=["txt", "csv", "tsv", "gz"],
    help="Expected columns: GENEID and NORMALIZEDEXPRESSIONVALUE"
)


# =========================================================
# AGE INPUT
# =========================================================

chronological_age = st.number_input(
    "Chronological age (optional)",
    min_value=0.0,
    max_value=120.0,
    value=0.0,
    step=1.0
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    st.success(
        f"File uploaded: {uploaded_file.name}"
    )

    try:

        # -------------------------------------------------
        # READ FILE
        # -------------------------------------------------

        if uploaded_file.name.endswith(".gz"):

            df = pd.read_csv(
                uploaded_file,
                sep="\t",
                compression="gzip",
                skiprows=6
            )

        elif uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_csv(
                uploaded_file,
                sep="\t",
                skiprows=6
            )


        # -------------------------------------------------
        # CHECK COLUMNS
        # -------------------------------------------------

        required_columns = [
            "GENEID",
            "NORMALIZEDEXPRESSIONVALUE"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

            st.info(
                "The file should contain GENEID and "
                "NORMALIZEDEXPRESSIONVALUE."
            )

            st.stop()


        # -------------------------------------------------
        # CLEAN DATA
        # -------------------------------------------------

        expression = (
            df[
                [
                    "GENEID",
                    "NORMALIZEDEXPRESSIONVALUE"
                ]
            ]
            .copy()
        )

        expression["GENEID"] = (
            expression["GENEID"]
            .astype(str)
        )

        expression[
            "NORMALIZEDEXPRESSIONVALUE"
        ] = pd.to_numeric(
            expression[
                "NORMALIZEDEXPRESSIONVALUE"
            ],
            errors="coerce"
        )

        expression = expression.dropna()


        # -------------------------------------------------
        # HANDLE DUPLICATE PROBES
        # -------------------------------------------------

        expression = (
            expression
            .groupby("GENEID")[
                "NORMALIZEDEXPRESSIONVALUE"
            ]
            .mean()
        )


        # -------------------------------------------------
        # CREATE MODEL INPUT
        # -------------------------------------------------

        sample = pd.DataFrame(
            index=[0],
            columns=feature_names
        )

        sample.iloc[0] = np.nan

        matched = 0

        for gene_id, value in expression.items():

            if gene_id in sample.columns:

                sample.loc[0, gene_id] = value

                matched += 1


        # Missing probes

        missing_count = sample.isna().sum().sum()

        # Model pipeline cannot accept NaN,
        # so use training-compatible median fallback.
        #
        # For this demo, unmatched probes are filled
        # using the median expression value of the uploaded sample.

        sample = sample.astype(float)

        sample = sample.fillna(
            expression.median()
        )


        # -------------------------------------------------
        # PREDICTION BUTTON
        # -------------------------------------------------

        st.write("")

        if st.button(
            "🔬 Predict Molecular Age",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing gene-expression profile..."
            ):

                predicted_age = float(
                    model.predict(sample)[0]
                )


            # -------------------------------------------------
            # RESULTS
            # -------------------------------------------------

            st.divider()

            st.subheader("🧬 AgeClock AI Result")


            col1, col2, col3 = st.columns(3)


            with col1:

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div class="small-text">
                    Predicted Age
                    </div>

                    <div class="result-number">
                    {predicted_age:.1f}
                    </div>

                    <div class="small-text">
                    years
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col2:

                if chronological_age > 0:

                    acceleration = (
                        predicted_age
                        - chronological_age
                    )

                    st.markdown(
                        f"""
                        <div class="result-card">

                        <div class="small-text">
                        Age Acceleration
                        </div>

                        <div class="result-number">
                        {acceleration:+.1f}
                        </div>

                        <div class="small-text">
                        predicted − chronological
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="result-card">

                        <div class="small-text">
                        Age Acceleration
                        </div>

                        <div class="result-number">
                        —
                        </div>

                        <div class="small-text">
                        Enter chronological age
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            with col3:

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div class="small-text">
                    Matched Probes
                    </div>

                    <div class="result-number">
                    {matched}
                    </div>

                    <div class="small-text">
                    of {len(feature_names)}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # -------------------------------------------------
            # INPUT QUALITY
            # -------------------------------------------------

            st.subheader("🔎 Input Quality")

            coverage = (
                matched / len(feature_names)
            ) * 100

            st.progress(
                min(coverage / 100, 1.0)
            )

            st.write(
                f"Feature coverage: **{coverage:.1f}%**"
            )


            # -------------------------------------------------
            # AGE COMPARISON
            # -------------------------------------------------

            if chronological_age > 0:

                st.subheader(
                    "📊 Chronological vs Predicted Age"
                )

                fig, ax = plt.subplots(
                    figsize=(8, 4)
                )

                ax.bar(
                    ["Chronological Age", "Predicted Age"],
                    [
                        chronological_age,
                        predicted_age
                    ]
                )

                ax.set_ylabel("Age (years)")

                ax.set_title(
                    "AgeClock AI Prediction"
                )

                st.pyplot(fig)


            # -------------------------------------------------
            # MODEL INFORMATION
            # -------------------------------------------------

            st.divider()

            st.subheader("📈 About the Model")

            info1, info2, info3 = st.columns(3)

            with info1:

                st.metric(
                    "Training Samples",
                    "54"
                )

            with info2:

                st.metric(
                    "Twin Pairs",
                    "27"
                )

            with info3:

                st.metric(
                    "Selected Probes",
                    "100"
                )


            st.info(
                """
                The current prototype was trained using
                gene-expression measurements from GSE74937.
                Because the dataset is small, predictions should
                be interpreted as research results rather than
                clinical measurements.
                """
            )


            # -------------------------------------------------
            # SELECTED FEATURES
            # -------------------------------------------------

            st.subheader(
                "🧬 Selected Molecular Probes"
            )

            selected_path = os.path.join(
                BASE_DIR,
                "model",
                "age_clock_selected_probes.csv"
            )

            if os.path.exists(selected_path):

                selected_df = pd.read_csv(
                    selected_path
                )

                st.dataframe(
                    selected_df.head(20),
                    use_container_width=True
                )


    except Exception as e:

        st.error(
            "Could not process the uploaded file."
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AgeClock AI • Research / Educational Prototype • "
    "Not a medical diagnostic system"
)