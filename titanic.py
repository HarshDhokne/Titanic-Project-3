from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "Titanic-Dataset.csv"
FEATURE_COLUMNS = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "family_size",
    "Isalone",
    "Sex_male",
    "Embarked_Q",
    "Embarked_S",
]

st.set_page_config(
    page_title="Titanic — Survival Studio",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    :root {
        --ink: #172b3a;
        --muted: #71818c;
        --sea: #123c4a;
        --teal: #2f827e;
        --sand: #f4efe5;
        --line: #e7e9e5;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 86% 0%, rgba(76, 143, 135, .10), transparent 31rem),
            #fbfaf7;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #123c4a;
        border-right: 0;
    }
    [data-testid="stSidebar"] * { color: #f5f2e9; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #c5d4d2;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }
    .eyebrow {
        color: var(--teal);
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }
    .hero-title {
        color: var(--ink);
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(2.5rem, 5vw, 4rem);
        line-height: 1.08;
        margin: 0;
    }
    .hero-copy {
        color: var(--muted);
        font-size: 1.04rem;
        line-height: 1.7;
        margin: .85rem 0 1.5rem;
        max-width: 690px;
    }
    .section-title {
        color: var(--ink);
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.55rem;
        margin: .25rem 0 .25rem;
    }
    .section-copy { color: var(--muted); margin: 0 0 1rem; }
    .metric-card {
        background: rgba(255, 255, 255, .82);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        min-height: 102px;
        box-shadow: 0 8px 24px rgba(21, 44, 54, .035);
    }
    .metric-label {
        color: var(--muted);
        font-size: .78rem;
        font-weight: 600;
        letter-spacing: .04em;
        text-transform: uppercase;
    }
    .metric-value {
        color: var(--ink);
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: .2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #123c4a, #205e63);
        border-radius: 20px;
        color: #fff;
        margin-top: 1.35rem;
        padding: 1.5rem;
    }
    .result-card h3 { color: #fff; margin: 0 0 .4rem; }
    .result-card p { color: #d2e2df; margin: 0; }
    .stButton > button {
        background: #2f827e;
        border: 0;
        border-radius: 10px;
        color: #fff;
        font-weight: 700;
        min-height: 3rem;
        transition: background .15s ease, transform .15s ease;
    }
    .stButton > button:hover {
        background: #246b68;
        border: 0;
        color: #fff;
        transform: translateY(-1px);
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem;
    }
    div[data-baseweb="tab-list"] { gap: .55rem; }
    button[data-baseweb="tab"] { border-radius: 9px 9px 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_passengers():
    return pd.read_csv(DATA_PATH)


if not DATA_PATH.is_file():
    st.error(f"Passenger dataset not found: {DATA_PATH.name}.")
    st.stop()

passengers = load_passengers()
prepared_passengers = passengers.copy()
prepared_passengers["Age"] = prepared_passengers["Age"].fillna(
    prepared_passengers["Age"].mean()
)
prepared_passengers["Embarked"] = prepared_passengers["Embarked"].fillna(
    prepared_passengers["Embarked"].mode().iloc[0]
)
prepared_passengers["family_size"] = (
    prepared_passengers["SibSp"] + prepared_passengers["Parch"] + 1
)
prepared_passengers["Isalone"] = (
    prepared_passengers["family_size"] == 1
).astype(int)
prepared_passengers = pd.get_dummies(
    prepared_passengers,
    columns=["Sex", "Embarked"],
    drop_first=True,
    dtype=int,
)
features = prepared_passengers.reindex(columns=FEATURE_COLUMNS)
labels = prepared_passengers["Survived"].astype(int)


@st.cache_resource
def train_models(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
        ),
        "Linear SVM": make_pipeline(
            StandardScaler(), SVC(kernel="linear", probability=True, random_state=42)
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=3)
        ),
    }
    metrics = []
    predictions = {}
    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        predicted = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        predictions[name] = predicted
        metrics.append(
            {
                "Algorithm": name,
                "Accuracy": accuracy_score(y_test, predicted),
                "Precision": precision_score(y_test, predicted, zero_division=0),
                "Recall": recall_score(y_test, predicted, zero_division=0),
                "F1 score": f1_score(y_test, predicted, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, probabilities),
            }
        )
    return models, pd.DataFrame(metrics).set_index("Algorithm"), y_test, predictions


models, model_metrics, y_test, test_predictions = train_models(features, labels)

with st.sidebar:
    st.markdown("## ⚓  SURVIVAL STUDIO")
    st.markdown("---")
    st.markdown("### A voyage through the data")
    st.markdown(
        "Explore the people aboard the Titanic and estimate a passenger's "
        "chance of survival using three machine-learning algorithms."
    )
    st.markdown("---")
    st.caption("MODELS")
    st.markdown("Logistic Regression · Linear SVM · KNN")
    st.caption("DATA")
    st.markdown(f"{len(passengers):,} passenger records")
    st.markdown("---")
    st.caption("A historical dataset, explored with care.")

st.markdown(
    f'<div class="eyebrow">A story of {len(passengers):,} lives in the manifest</div>',
    unsafe_allow_html=True,
)
st.markdown('<h1 class="hero-title">The voyage, in numbers.</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-copy">Step aboard the Titanic dataset. Explore the passenger '
    'manifest, compare three survival models, then try a passenger estimate.</p>',
    unsafe_allow_html=True,
)

total_passengers = len(passengers)
survival_rate = passengers["Survived"].mean() * 100
female_survival = passengers.loc[passengers["Sex"] == "female", "Survived"].mean() * 100
male_survival = passengers.loc[passengers["Sex"] == "male", "Survived"].mean() * 100

metric_columns = st.columns(4)
metrics = [
    ("Passengers", f"{total_passengers:,}"),
    ("Survived", f"{int(passengers['Survived'].sum()):,}"),
    ("Overall survival", f"{survival_rate:.1f}%"),
    ("Women survived", f"{female_survival:.1f}%"),
]
for column, (label, value) in zip(metric_columns, metrics):
    with column:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

prediction_tab, insights_tab, comparison_tab = st.tabs(
    ["Passenger estimate", "Manifest insights", "Model comparison"]
)

with prediction_tab:
    st.markdown('<div class="section-title">Build a passenger profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Choose an algorithm and adjust the details below to see its prediction.</p>',
        unsafe_allow_html=True,
    )

    with st.form("passenger_form"):
        selected_model = st.selectbox("Algorithm", list(models))
        left, right = st.columns(2, gap="large")
        with left:
            passenger_class = st.selectbox(
                "Passenger class",
                options=[1, 2, 3],
                index=2,
                format_func=lambda value: {
                    1: "1st class — upper deck",
                    2: "2nd class — middle deck",
                    3: "3rd class — lower deck",
                }[value],
            )
            age = st.slider("Age", min_value=0, max_value=80, value=29)
            sex = st.radio("Sex recorded in manifest", ["Female", "Male"], horizontal=True)
        with right:
            fare = st.number_input(
                "Fare paid (£)",
                min_value=0.0,
                max_value=600.0,
                value=32.0,
                step=1.0,
            )
            siblings_spouses = st.number_input(
                "Siblings / spouses aboard",
                min_value=0,
                max_value=8,
                value=0,
                step=1,
            )
            parents_children = st.number_input(
                "Parents / children aboard",
                min_value=0,
                max_value=6,
                value=0,
                step=1,
            )
            embarked = st.selectbox(
                "Port of embarkation",
                options=["Southampton", "Cherbourg", "Queenstown"],
                format_func=lambda value: {
                    "Southampton": "Southampton (S)",
                    "Cherbourg": "Cherbourg (C)",
                    "Queenstown": "Queenstown (Q)",
                }[value],
            )

        submitted = st.form_submit_button("Estimate survival", use_container_width=True)

    if submitted:
        family_size = int(siblings_spouses + parents_children + 1)
        passenger_features = pd.DataFrame(
            [[
                passenger_class,
                age,
                siblings_spouses,
                parents_children,
                fare,
                family_size,
                int(family_size == 1),
                int(sex == "Male"),
                int(embarked == "Queenstown"),
                int(embarked == "Southampton"),
            ]],
            columns=FEATURE_COLUMNS,
        )
        estimator = models[selected_model]
        survived = int(estimator.predict(passenger_features)[0])
        probabilities = estimator.predict_proba(passenger_features)[0]
        survival_probability = float(
            probabilities[list(estimator.classes_).index(1)]
        )

        if survived:
            result_title = f"{selected_model} predicts survival."
            result_copy = "This profile is more likely to survive according to the fitted model."
            st.success(f"Estimated survival likelihood: {survival_probability:.1%}")
        else:
            result_title = f"{selected_model} predicts non-survival."
            result_copy = "This profile is less likely to survive according to the fitted model."
            st.warning(f"Estimated survival likelihood: {survival_probability:.1%}")

        st.markdown(
            f'<div class="result-card"><h3>{result_title}</h3>'
            f'<p>{result_copy} This is a historical model estimate, not a certainty.</p></div>',
            unsafe_allow_html=True,
        )

with insights_tab:
    st.markdown('<div class="section-title">The manifest at a glance</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Survival varied across the passenger manifest. '
        'These views summarize the records included with this project.</p>',
        unsafe_allow_html=True,
    )
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.markdown("#### Survival by passenger class")
        class_rates = (
            passengers.groupby("Pclass")["Survived"]
            .mean()
            .mul(100)
            .rename("Survival rate (%)")
            .to_frame()
        )
        class_rates.index = [f"{value}{'st' if value == 1 else 'nd' if value == 2 else 'rd'} class" for value in class_rates.index]
        st.bar_chart(class_rates, color="#2f827e", y_label="Survival rate (%)")
    with chart_right:
        st.markdown("#### Survival by sex")
        sex_rates = (
            passengers.groupby("Sex")["Survived"]
            .mean()
            .mul(100)
            .rename("Survival rate (%)")
            .to_frame()
        )
        st.bar_chart(sex_rates, color="#c58b58", y_label="Survival rate (%)")

    st.markdown("#### A closer look")
    insight_columns = st.columns(3)
    insight_columns[0].metric("Women survived", f"{female_survival:.1f}%")
    insight_columns[1].metric("Men survived", f"{male_survival:.1f}%")
    insight_columns[2].metric("Average age", f"{passengers['Age'].mean():.1f} years")
    st.caption(
        "Charts describe this dataset only. The models were trained on "
        "passenger class, age, fare, family size, sex, and embarkation port. "
        "All three algorithms are compared on the same stratified 20% holdout set."
    )

with comparison_tab:
    st.markdown(
        '<div class="section-title">Compare the three algorithms</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-copy">Each model uses the same stratified 80/20 train-test '
        'split (random state 42) and standardized features.</p>',
        unsafe_allow_html=True,
    )
    model_columns = st.columns(len(models))
    for column, (name, scores) in zip(model_columns, model_metrics.iterrows()):
        with column:
            st.markdown(f"#### {name}")
            for metric, score in scores.items():
                st.progress(float(score), text=f"{metric}: {score:.1%}")

    st.dataframe(
        model_metrics.style.format("{:.1%}"),
        width="stretch",
    )

    st.markdown("#### Confusion matrices")
    confusion_columns = st.columns(len(models))
    for column, (name, predicted) in zip(confusion_columns, test_predictions.items()):
        with column:
            st.markdown(f"**{name}**")
            matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["Actual: did not survive", "Actual: survived"],
                    columns=["Predicted: did not survive", "Predicted: survived"],
                ),
                width="stretch",
            )
