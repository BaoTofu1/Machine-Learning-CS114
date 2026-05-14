"""
Bank Marketing Prediction Web Demo
This app demonstrates machine learning models for predicting bank marketing campaign success.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    balanced_accuracy_score,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
from pathlib import Path

# Optional dependency: SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

# Optional dependency: LightGBM
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
DATA_PATH_CANDIDATES = [
    APP_DIR / "Models" / "processed_bank_data.csv",
    APP_DIR.parent / "Models" / "processed_bank_data.csv",
    Path.cwd() / "Models" / "processed_bank_data.csv",
    Path.cwd().parent / "Models" / "processed_bank_data.csv",
]

# Model options used across pages
MODEL_OPTIONS = [
    "Logistic Regression",
    "XGBoost", 
    "Random Forest (RF)",
    "LightGBM",
]
MODEL_HELP_TEXT = f"Choose a model: {', '.join(MODEL_OPTIONS)}"

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {padding: 0rem 0rem;}
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# LOAD AND CACHE DATA
# ============================================================================
@st.cache_data
def load_data():
    """Load the bank marketing dataset"""
    for data_path in DATA_PATH_CANDIDATES:
        if data_path.exists():
            return pd.read_csv(data_path)

    searched_paths = "\n".join(f"- {path}" for path in DATA_PATH_CANDIDATES)
    st.error(
        "Data file not found. Expected processed_bank_data.csv in one of these locations:\n"
        f"{searched_paths}"
    )
    return None


@st.cache_resource
def prepare_data(df):
    """Prepare data for modeling"""
    # Feature engineering
    df_processed = df.copy()
    df_processed["is_contacted_before"] = (df_processed["pdays"] != 999).astype(int)

    X = df_processed.drop(columns=["y"])
    y = df_processed["y"]

    # Time split (no shuffle for temporal consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    return X, y, X_train, X_test, y_train, y_test, X.columns


@st.cache_resource
def process_scenario(X_tr, X_te, y_tr, apply_smote=True):
    """Process data for a specific scenario (with/without duration)"""
    X_tr = X_tr.astype(float)
    X_te = X_te.astype(float)

    scaler = StandardScaler()
    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_te_scaled = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)

    if apply_smote:
        smote = SMOTE(random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_tr)
    else:
        X_tr_res, y_tr_res = X_tr_scaled, y_tr

    return X_tr_res, y_tr_res, X_te_scaled, scaler


@st.cache_resource
def train_models(X_train, X_test, y_train, y_test, model_type="Logistic Regression"):
    """Train models for both scenarios using specified algorithm"""
    results = {}

    # Scenario 1: With duration (Benchmark)
    X_tr1 = X_train.copy()
    X_te1 = X_test.copy()
    X_tr1_res, y_tr1_res, X_te1_scaled, scaler1 = process_scenario(X_tr1, X_te1, y_train)

    if model_type == "Logistic Regression":
        model1 = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "XGBoost":
        model1 = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.03,
            subsample=1.0,
            colsample_bytree=1.0,
            min_child_weight=1,
            reg_lambda=2.0,
            random_state=42,
            objective="binary:logistic",
            eval_metric='logloss',
            n_jobs=-1
        )
    elif model_type == "Random Forest (RF)":
        model1 = RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            max_features=0.5,
            min_samples_leaf=2,
            min_samples_split=16,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "LightGBM" and LIGHTGBM_AVAILABLE:
        model1 = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=1.0,
            reg_lambda=0.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        # Fallback to Logistic Regression
        model1 = LogisticRegression(max_iter=1000, random_state=42)
    
    model1.fit(X_tr1_res, y_tr1_res)

    y_pred1 = model1.predict(X_te1_scaled)
    y_pred_proba1 = model1.predict_proba(X_te1_scaled)[:, 1]

    results["scenario_1"] = {
        "model": model1,
        "scaler": scaler1,
        "X_train": X_tr1_res,
        "X_test": X_te1_scaled,
        "y_pred": y_pred1,
        "y_pred_proba": y_pred_proba1,
        "features": X_tr1.columns,
        "name": "With Duration (Benchmark)",
        "model_type": model_type,
    }

    # Scenario 2: Without duration (Realistic)
    X_tr2 = X_train.drop(columns=["duration"])
    X_te2 = X_test.drop(columns=["duration"])
    X_tr2_res, y_tr2_res, X_te2_scaled, scaler2 = process_scenario(X_tr2, X_te2, y_train)

    if model_type == "Logistic Regression":
        model2 = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "XGBoost":
        model2 = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.12,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_lambda=2.0,
            random_state=42,
            objective="binary:logistic",
            eval_metric='logloss',
            n_jobs=-1
        )
    elif model_type == "Random Forest (RF)":
        model2 = RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features=0.5,
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "LightGBM":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Please install lightgbm to use this model.")
        model2 = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=1.0,
            reg_lambda=0.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model2.fit(X_tr2_res, y_tr2_res)

    y_pred2 = model2.predict(X_te2_scaled)
    y_pred_proba2 = model2.predict_proba(X_te2_scaled)[:, 1]

    results["scenario_2"] = {
        "model": model2,
        "scaler": scaler2,
        "X_train": X_tr2_res,
        "X_test": X_te2_scaled,
        "y_pred": y_pred2,
        "y_pred_proba": y_pred_proba2,
        "features": X_tr2.columns,
        "name": "Without Duration",
        "model_type": model_type,
    }

    return results, y_test

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate classification metrics"""
    return {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "f1": f1_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def get_local_explanation(result, input_df, input_scaled):
    """Return per-prediction feature contributions."""
    model = result["model"]
    model_type = result["model_type"]
    features = list(result["features"])
    x_scaled = np.asarray(input_scaled).reshape(1, -1)
    x_raw = np.asarray(input_df).reshape(1, -1)

    method = ""
    base_value = None

    if model_type == "Logistic Regression":
        # Local contribution in log-odds space for linear model
        shap_values = x_scaled[0] * model.coef_[0]
        base_value = float(model.intercept_[0])
        method = "Linear contribution (coefficient × scaled feature value)"
    else:
        try:
            if not SHAP_AVAILABLE:
                raise ImportError("SHAP is not installed")
            explainer = shap.TreeExplainer(model)
            shap_output = explainer.shap_values(x_scaled)
            if isinstance(shap_output, list):
                shap_values = np.asarray(shap_output[-1]).reshape(-1)
            else:
                shap_values = np.asarray(shap_output).reshape(-1)

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                base_value = float(np.asarray(expected_value).reshape(-1)[-1])
            else:
                base_value = float(expected_value)
            method = "SHAP TreeExplainer"
        except Exception:
            # Fallback if SHAP fails for environment/package mismatch
            importances = getattr(model, "feature_importances_", np.ones(x_scaled.shape[1]))
            shap_values = np.asarray(importances).reshape(-1) * x_scaled[0]
            method = "Approximate contribution (feature_importance × scaled value)"

    explanation_df = pd.DataFrame(
        {
            "Feature": features,
            "Input Value": x_raw[0],
            "Scaled Value": x_scaled[0],
            "Contribution": shap_values,
            "Abs Contribution": np.abs(shap_values),
        }
    ).sort_values("Abs Contribution", ascending=False)

    return explanation_df, method, base_value


def ensure_model_available(model_type):
    """Gracefully handle optional model dependencies"""
    if model_type == "LightGBM" and not LIGHTGBM_AVAILABLE:
        st.error("LightGBM is not installed in this environment. Install it with: `pip install lightgbm`")
        st.stop()


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("🏦 Bank Marketing Demo")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Data Exploration", "Model Training", "Make Predictions", "Model Comparison"],
    label_visibility="visible",
)
if not LIGHTGBM_AVAILABLE:
    st.sidebar.caption("LightGBM not available in this environment (`pip install lightgbm`).")

# ============================================================================
# LOAD DATA
# ============================================================================
df = load_data()

if df is None:
    st.stop()

X, y, X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

# ============================================================================
# PAGE: HOME
# ============================================================================
if page == "Home":
    st.markdown(
        """
    # 🏦 Bank Marketing Campaign Prediction
    
    ## Project Overview
    This project uses Machine Learning to predict whether a customer will subscribe 
    to a bank product based on their demographic and campaign information.
    
    ### 📊 Dataset Information
    """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Number of Features", len(df.columns) - 1)
    col3.metric("Subscription Rate", f"{y.mean():.2%}")
    col4.metric("Class Balance", f"Class 1: {y.sum():,} | Class 0: {len(y) - y.sum():,}")

    st.markdown(
        """
    ### 🎯 Project Goals
    
    1. **Exploratory Data Analysis (EDA)**: Understand the characteristics of customers 
       who subscribe vs. those who don't
    2. **Feature Engineering**: Create meaningful features like `is_contacted_before`
    3. **Model Development**: Train and compare multiple machine learning models
    4. **Two Scenarios**:
       - **Scenario 1 (Benchmark)**: Include `duration` feature for performance ceiling
       - **Scenario 2 (Realistic)**: Exclude `duration` to simulate real-world predictions
    
    ### 🔧 Key Techniques
    - **Preprocessing**: StandardScaler for numerical features
    - **Imbalanced Data Handling**: SMOTE for data balancing
    - **Time Split**: Train/test split without shuffling to respect temporal patterns
    - **Models**: Logistic Regression and ensemble methods
    
    ### 📈 Key Insights
    
    **Data Characteristics:**
    - The target variable is highly imbalanced (only ~11% subscriptions)
    - Duration feature shows data leakage - customers who talk longer are more likely to subscribe
    - Time drift is present - model performance drops when applied to future data
    - Without duration, ROC-AUC drops to ~0.64, showing realistic model performance
    
    **Preprocessing Strategy:**
    - SMOTE oversamples the minority class to balance the dataset
    - Time-based train/test split preserves temporal order
    - Standardization is essential for Logistic Regression
    
    ### 🚀 How to Use This Demo
    
    1. **Data Exploration**: Visualize and understand the data distribution
    2. **Model Training**: See how the models are trained with different scenarios
    3. **Make Predictions**: Input customer data to get predictions
    4. **Model Comparison**: Compare performance across scenarios and models
    """
    )

    st.divider()
    st.markdown("### 📚 Dataset Features")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numerical Features:**")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols[:len(numerical_cols)//2]:
            st.write(f"- {col}")

    with col2:
        st.markdown("**Categorical Features:**")
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            st.write(f"- {col}")


# ============================================================================
# PAGE: DATA EXPLORATION
# ============================================================================
elif page == "Data Exploration":
    st.title("📊 Data Exploration & Analysis")
 
    # ─── 1. Dataset Overview ────────────────────────────────────────────────
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    col4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
 
    with st.expander("📄 View Raw Data"):
        st.dataframe(df.head(20), use_container_width=True)
 
    # ─── 2. Missing Values ──────────────────────────────────────────────────
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing (%)": missing_pct
    }).query("`Missing Count` > 0")
 
    if missing_df.empty:
        st.success("✅ No missing values in the dataset!")
    else:
        st.warning(f"⚠️ {len(missing_df)} column(s) have missing values.")
        st.dataframe(missing_df, use_container_width=True)
 
    # ─── 3. Statistical Summary ─────────────────────────────────────────────
    st.subheader("Statistical Summary")
    tab_num, tab_cat = st.tabs(["Numerical", "Categorical"])
 
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "y" in numerical_cols:
        numerical_cols.remove("y")
 
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
 
    with tab_num:
        st.dataframe(df[numerical_cols].describe().T.style.format("{:.2f}"), use_container_width=True)
 
    with tab_cat:
        if cat_cols:
            cat_summary = pd.DataFrame({
                "Unique Values": df[cat_cols].nunique(),
                "Most Frequent": df[cat_cols].mode().iloc[0],
                "Frequency": [df[c].value_counts().iloc[0] for c in cat_cols],
            })
            st.dataframe(cat_summary, use_container_width=True)
        else:
            st.info("No categorical columns found.")
 
    # ─── 4. Target Distribution ─────────────────────────────────────────────
    st.subheader("Target Variable Distribution")
    target_counts = y.value_counts()
    colors = ["#FF6B6B", "#4ECDC4"]
 
    col1, col2 = st.columns(2)
 
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(
            ["No (0)", "Yes (1)"],
            target_counts.values,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Target Distribution", fontsize=14, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, v in zip(bars, target_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + len(y) * 0.01,
                f"{v:,}\n({v / len(y) * 100:.1f}%)",
                ha="center",
                fontweight="bold",
                fontsize=10,
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
 
    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sizes = target_counts.values
        labels = [f"No  ({sizes[0]:,})", f"Yes  ({sizes[1]:,})"]
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for at in autotexts:
            at.set_fontweight("bold")
        ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
 
    # ─── 5. Outlier Detection ───────────────────────────────────────────────
    st.subheader("Outlier Detection (IQR Method)")
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    outlier_counts = (
        (df[numerical_cols] < (Q1 - 1.5 * IQR)) |
        (df[numerical_cols] > (Q3 + 1.5 * IQR))
    ).sum()
    outlier_df = pd.DataFrame({
        "Outlier Count": outlier_counts,
        "Outlier (%)": (outlier_counts / len(df) * 100).round(2),
    }).query("`Outlier Count` > 0").sort_values("Outlier Count", ascending=False)
 
    if outlier_df.empty:
        st.success("✅ No significant outliers detected!")
    else:
        st.dataframe(outlier_df, use_container_width=True)
 
    # ─── 6. Numerical Feature Distribution ──────────────────────────────────
    st.subheader("Numerical Features Distribution")
    selected_features = st.multiselect(
        "Select numerical features to visualize:",
        numerical_cols,
        default=numerical_cols[:6],
        max_selections=12,
    )
 
    if selected_features:
        n_cols = 2
        n_rows = (len(selected_features) + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)
        axes_flat = axes.flatten()
 
        for idx, col in enumerate(selected_features):
            ax = axes_flat[idx]
            ax.hist(df[col], bins=30, color="steelblue", alpha=0.75, edgecolor="black", linewidth=0.5)
            ax.set_title(f"{col}", fontweight="bold", fontsize=11)
            ax.set_xlabel(col, fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
 
            # Add mean & median lines
            ax.axvline(df[col].mean(), color="#FF6B6B", linestyle="--", linewidth=1.5, label=f"Mean: {df[col].mean():.2f}")
            ax.axvline(df[col].median(), color="#4ECDC4", linestyle="--", linewidth=1.5, label=f"Median: {df[col].median():.2f}")
            ax.legend(fontsize=8)
 
        for idx in range(len(selected_features), len(axes_flat)):
            axes_flat[idx].axis("off")
 
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
 
    # ─── 7. Feature vs Target ───────────────────────────────────────────────
    st.subheader("Feature vs Target Variable")
    col1, col2 = st.columns([1, 2])
 
    with col1:
        selected_vs_target = st.selectbox("Select feature to compare with target:", numerical_cols)
 
    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
        # Boxplot
        groups = [df[selected_vs_target][y == val] for val in sorted(y.unique())]
        axes[0].boxplot(groups, labels=["No (0)", "Yes (1)"], patch_artist=True,
                        boxprops=dict(facecolor="steelblue", alpha=0.6),
                        medianprops=dict(color="#FF6B6B", linewidth=2))
        axes[0].set_title(f"{selected_vs_target} by Target", fontweight="bold")
        axes[0].spines[["top", "right"]].set_visible(False)
 
        # KDE overlay
        for val, color, label in zip(sorted(y.unique()), colors, ["No (0)", "Yes (1)"]):
            subset = df[selected_vs_target][y == val].dropna()
            axes[1].hist(subset, bins=30, alpha=0.5, color=color, label=label, density=True)
        axes[1].set_title(f"Distribution by Target", fontweight="bold")
        axes[1].legend()
        axes[1].spines[["top", "right"]].set_visible(False)
 
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
 
    # ─── 8. Categorical Feature Analysis ────────────────────────────────────
    if cat_cols:
        st.subheader("Categorical Features Analysis")
        selected_cat = st.selectbox("Select categorical feature:", cat_cols)
 
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
        # Value counts bar
        vc = df[selected_cat].value_counts()
        axes[0].barh(vc.index.astype(str), vc.values, color="steelblue", alpha=0.8, edgecolor="black")
        axes[0].set_title(f"Value Counts: {selected_cat}", fontweight="bold")
        axes[0].invert_yaxis()
        axes[0].spines[["top", "right"]].set_visible(False)
 
        # Stacked bar vs target
        cross = pd.crosstab(df[selected_cat], y, normalize="index") * 100
        cross.plot(kind="bar", stacked=True, ax=axes[1], color=colors, alpha=0.85, edgecolor="black")
        axes[1].set_title(f"{selected_cat} vs Target (%)", fontweight="bold")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Percentage (%)")
        axes[1].legend(["No (0)", "Yes (1)"], loc="upper right")
        axes[1].spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=30, ha="right")
 
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
 
    # ─── 9. Correlation Analysis ─────────────────────────────────────────────
    st.subheader("Feature Correlation Analysis")
 
    corr_features = st.multiselect(
        "Select features for correlation matrix:",
        numerical_cols,
        default=numerical_cols[:12],
        max_selections=20,
    )
 
    if corr_features:
        corr_matrix = df[corr_features].corr()
 
        fig, ax = plt.subplots(figsize=(max(10, len(corr_features)), max(8, len(corr_features) - 1)))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # upper triangle mask
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            mask=mask,
            linewidths=0.5,
            ax=ax,
            annot_kws={"size": 8},
        )
        ax.set_title("Correlation Matrix (Lower Triangle)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
 
        # Top correlated pairs
        corr_pairs = (
            corr_matrix.where(mask)
            .stack()
            .reset_index()
        )
        corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
        corr_pairs["Abs Correlation"] = corr_pairs["Correlation"].abs()
        # Get top 10 pairs with highest absolute correlation (not including self-correlations)
        top_pairs = corr_pairs[corr_pairs["Feature 1"] != corr_pairs["Feature 2"]].sort_values("Abs Correlation", ascending=False).head(10)
 
        with st.expander("🔍 Top 10 Correlated Feature Pairs"):
            st.dataframe(
                top_pairs[["Feature 1", "Feature 2", "Correlation"]].style.background_gradient(
                    cmap="coolwarm", subset=["Correlation"]
                ).format({"Correlation": "{:.4f}"}),
                use_container_width=True,
            )
 
        # Download button
        csv = corr_matrix.to_csv().encode("utf-8")
        st.download_button(
            label="⬇️ Download Correlation Matrix (CSV)",
            data=csv,
            file_name="correlation_matrix.csv",
            mime="text/csv",
        )

# ============================================================================
# PAGE: MODEL TRAINING
# ============================================================================
elif page == "Model Training":
    st.title("🤖 Model Training & Evaluation")

    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Select Model:",
            MODEL_OPTIONS,
            help=MODEL_HELP_TEXT
        )
    
    with col2:
        scenario = st.radio("Select Scenario:", ["Scenario 1: With Duration (Benchmark)", 
                                              "Scenario 2: Without Duration (Realistic)"])
    scenario_key = "scenario_1" if "1" in scenario else "scenario_2"
    ensure_model_available(model_type)

    # Train models
    with st.spinner(f"Training {model_type} models..."):
        models, y_test_data = train_models(X_train, X_test, y_train, y_test, model_type)

    result = models[scenario_key]
    y_pred = result["y_pred"]
    y_pred_proba = result["y_pred_proba"]
    metrics = calculate_metrics(y_test_data, y_pred, y_pred_proba)

    st.subheader(f"📊 {result['name']}")

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    col2.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
    col3.metric("F1-Score", f"{metrics['f1']:.4f}")
    col4.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}")

    # Classification report
    st.subheader("Classification Report")
    report_dict = classification_report(
        y_test_data, y_pred, output_dict=True, target_names=["No (0)", "Yes (1)"]
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test_data, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No (0)", "Yes (1)"],
            yticklabels=["No (0)", "Yes (1)"],
            ax=ax,
        )
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        st.pyplot(fig, use_container_width=True)

    with col2:
        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test_data, y_pred_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f'Model (AUC = {metrics["roc_auc"]:.4f})')
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        st.pyplot(fig, use_container_width=True)

    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")
    prec, rec, _ = precision_recall_curve(y_test_data, y_pred_proba)
    baseline = y_test_data.mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rec, prec, color="darkorange", lw=2, label=f'Model (AP = {metrics["pr_auc"]:.4f})')
    ax.axhline(y=baseline, color="navy", linestyle="--", lw=1, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    st.pyplot(fig, use_container_width=True)

    # Feature Importance
    if result["model_type"] == "Logistic Regression":
        st.subheader("Feature Importance (Model Coefficients)")
        feature_importance = pd.DataFrame(
            {
                "Feature": result["features"],
                "Coefficient": result["model"].coef_[0],
            }
        ).sort_values("Coefficient", key=abs, ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 5 Features Increasing Subscription Probability**")
            top_positive = feature_importance.sort_values("Coefficient", ascending=False).head(5)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(top_positive["Feature"], top_positive["Coefficient"], color="green", alpha=0.7)
            ax.set_xlabel("Coefficient Value", fontsize=12)
            ax.set_title("Positive Influence on Subscription", fontsize=12, fontweight="bold")
            ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.markdown("**Top 5 Features Decreasing Subscription Probability**")
            top_negative = feature_importance.sort_values("Coefficient", ascending=True).head(5)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(top_negative["Feature"], top_negative["Coefficient"], color="red", alpha=0.7)
            ax.set_xlabel("Coefficient Value", fontsize=12)
            ax.set_title("Negative Influence on Subscription", fontsize=12, fontweight="bold")
            ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
            st.pyplot(fig, use_container_width=True)

        st.dataframe(feature_importance, use_container_width=True)
    else:
        st.subheader(f"Feature Importance ({result['model_type']})")
        feature_importance = pd.DataFrame(
            {
                "Feature": result["features"],
                "Importance": result["model"].feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 10 Most Important Features**")
            top_features = feature_importance.head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_features["Feature"], top_features["Importance"], color="steelblue", alpha=0.7)
            ax.set_xlabel("Feature Importance", fontsize=12)
            ax.set_title(f"{result['model_type']} Feature Importance", fontsize=12, fontweight="bold")
            ax.invert_yaxis()
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.markdown("**Feature Importance Summary**")
            st.info(
                f"🔍 **{result['model_type']} Feature Importance**\n\n"
                "Tree-based models rank features by how strongly they reduce prediction error:\n"
                "- Higher values = more important for predictions\n"
                "- Values are relative importance scores\n"
                "- Different from Logistic Regression coefficients which show direction"
            )

        st.dataframe(feature_importance, use_container_width=True)


# ============================================================================
# PAGE: MAKE PREDICTIONS
# ============================================================================
elif page == "Make Predictions":
    st.title("🔮 Make Predictions for New Customers")

    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Select Model Algorithm:",
            MODEL_OPTIONS,
            help=MODEL_HELP_TEXT
        )
    
    with col2:
        scenario = st.radio(
            "Select Model Scenario:",
            ["Scenario 1: With Duration (Benchmark)", "Scenario 2: Without Duration (Realistic)"],
        )
    scenario_key = "scenario_1" if "1" in scenario else "scenario_2"
    ensure_model_available(model_type)

    # Train models
    with st.spinner(f"Loading {model_type} models..."):
        models, y_test_data = train_models(X_train, X_test, y_train, y_test, model_type)
    result = models[scenario_key]
    model = result["model"]
    scaler = result["scaler"]
    features_used = list(result["features"])

    st.subheader("📋 Input Customer Data")

    # Create input columns
    input_data = {}

    col1, col2 = st.columns(2)
    cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    num_features = ["duration", "age", "balance", "pdays", "previous"]
    hidden_cat_features = ["age_group"]
    
    # Group one-hot encoded categorical columns (e.g., job_blue-collar, marital_single)
    categorical_groups = {}
    for cat in cat_features:
        encoded_cols = sorted([f for f in features_used if f.startswith(f"{cat}_")])
        if encoded_cols:
            categorical_groups[cat] = encoded_cols

    # Render selectbox for each categorical feature group
    for idx, (cat, encoded_cols) in enumerate(categorical_groups.items()):
        if idx % 2 == 0:
            col = col1
        else:
            col = col2

        options = {}
        has_base_category = (df[encoded_cols].sum(axis=1) == 0).any()
        if has_base_category:
            options["None"] = "__base__"

        for encoded_col in encoded_cols:
            options[encoded_col.replace(f"{cat}_", "", 1)] = encoded_col

        selected_label = col.selectbox(f"{cat}", list(options.keys()))
        selected_encoded_col = options[selected_label]

        for encoded_col in encoded_cols:
            input_data[encoded_col] = 0
        if selected_encoded_col != "__base__":
            input_data[selected_encoded_col] = 1

    # Initialize hidden one-hot groups (not shown in UI)
    for hidden_cat in hidden_cat_features:
        hidden_cols = sorted([f for f in features_used if f.startswith(f"{hidden_cat}_")])
        if not hidden_cols:
            continue

        for encoded_col in hidden_cols:
            input_data[encoded_col] = 0

    handled_categorical_cols = set()
    for encoded_cols in categorical_groups.values():
        handled_categorical_cols.update(encoded_cols)
    for hidden_cat in hidden_cat_features:
        handled_categorical_cols.update([f for f in features_used if f.startswith(f"{hidden_cat}_")])

    remaining_features = [f for f in features_used if f not in handled_categorical_cols]

    # Keep all selectboxes above numeric sliders/inputs
    selectbox_features = [f for f in remaining_features if f == "is_contacted_before"]
    number_input_features = [f for f in remaining_features if f in num_features]
    slider_features = [
        f for f in remaining_features
        if f not in selectbox_features and f not in number_input_features
    ]

    render_idx = len(categorical_groups)

    for feature in selectbox_features:
        col = col1 if render_idx % 2 == 0 else col2
        input_data[feature] = col.selectbox(
            f"{feature}",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        render_idx += 1

    for feature in number_input_features:
        col = col1 if render_idx % 2 == 0 else col2
        input_data[feature] = col.number_input(
            f"{feature}",
            min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median()),
        )
        render_idx += 1

    for feature in slider_features:
        col = col1 if render_idx % 2 == 0 else col2
        input_data[feature] = col.slider(
            f"{feature}",
            min_value=float(df[feature].min()),
            max_value=float(df[feature].max()),
            value=float(df[feature].median()),
        )
        render_idx += 1

    # Derive age_group from age to keep inference consistent with preprocessing
    age_group_cols = sorted([f for f in features_used if f.startswith("age_group_")])
    if "age" in input_data and age_group_cols:
        for col_name in age_group_cols:
            input_data[col_name] = 0

        age_value = float(input_data["age"])
        if age_value <= 25:
            inferred_age_group = "young"
        elif age_value <= 40:
            inferred_age_group = "adult"
        elif age_value <= 60:
            inferred_age_group = "middle"
        else:
            inferred_age_group = "senior"

        inferred_age_col = f"age_group_{inferred_age_group}"
        if inferred_age_col in age_group_cols:
            input_data[inferred_age_col] = 1

    # Make prediction
    if st.button("🎯 Predict", use_container_width=True, type="primary"):
        # Prepare input
        input_df = pd.DataFrame([input_data])[features_used]
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        # Display results
        st.divider()
        st.subheader("🎉 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.success("✅ LIKELY TO SUBSCRIBE", icon="✅")
                st.metric("Subscription Probability", f"{probability[1]:.2%}")
            else:
                st.warning("❌ UNLIKELY TO SUBSCRIBE", icon="❌")
                st.metric("Subscription Probability", f"{probability[1]:.2%}")

        with col2:
            st.markdown("**Prediction Breakdown**")
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#FF6B6B", "#4ECDC4"]
            bars = ax.bar(["No Subscribe", "Subscribe"], probability, color=colors, alpha=0.7, edgecolor="black")
            ax.set_ylabel("Probability", fontsize=12)
            ax.set_title("Prediction Probabilities", fontsize=12, fontweight="bold")
            ax.set_ylim([0, 1])
            for bar, prob in zip(bars, probability):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{prob:.1%}", ha="center", va="bottom")
            st.pyplot(fig, use_container_width=True)

        # Local explanation for the current prediction
        st.subheader("🧠 Prediction Explanation")
        explanation_df, explanation_method, base_value = get_local_explanation(result, input_df, input_scaled)

        st.caption(f"Method: {explanation_method}")
        if base_value is not None:
            st.caption(f"Base value (model raw score baseline): {base_value:.4f}")

        top_n = 12
        top_explanations = explanation_df.head(top_n).iloc[::-1]
        contribution_colors = np.where(
            top_explanations["Contribution"] >= 0,
            "#2E8B57",
            "#C0392B",
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            top_explanations["Feature"],
            top_explanations["Contribution"],
            color=contribution_colors,
            alpha=0.85,
        )
        ax.axvline(x=0, color="black", linewidth=1)
        ax.set_xlabel("Feature Contribution", fontsize=12)
        ax.set_title("Top Local Feature Contributions", fontsize=13, fontweight="bold")
        st.pyplot(fig, use_container_width=True)

        st.dataframe(
            explanation_df[["Feature", "Input Value", "Contribution"]].head(top_n).round(4),
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("**Input Summary:**")
        st.json(input_data)


# ============================================================================
# PAGE: MODEL COMPARISON
# ============================================================================
elif page == "Model Comparison":
    st.title("📊 Model Comparison")

    # Scenario selector
    scenario = st.selectbox(
        "Select Scenario:",
        ["Scenario 1: With Duration (Benchmark)", "Scenario 2: Without Duration (Realistic)"],
        help="Choose which scenario to display model comparison"
    )
    scenario_key = "scenario_1" if "1" in scenario else "scenario_2"
    scenario_name = "With Duration" if "1" in scenario else "Without Duration"

    # Train all 4 models for comparison
    with st.spinner("Training all models for comparison..."):
        all_results = {}
        all_metrics = {}
        
        for model_type in MODEL_OPTIONS:
            ensure_model_available(model_type)
            models, y_test_data = train_models(X_train, X_test, y_train, y_test, model_type)
            result = models[scenario_key]
            metrics = calculate_metrics(y_test_data, result["y_pred"], result["y_pred_proba"])
            all_results[model_type] = result
            all_metrics[model_type] = metrics

    # Create comparison DataFrame
    comparison_data = {
        "Model": MODEL_OPTIONS,
        "ROC-AUC": [all_metrics[model]["roc_auc"] for model in MODEL_OPTIONS],
        "PR-AUC": [all_metrics[model]["pr_auc"] for model in MODEL_OPTIONS],
        "F1-Score": [all_metrics[model]["f1"] for model in MODEL_OPTIONS],
        "Balanced Accuracy": [all_metrics[model]["balanced_accuracy"] for model in MODEL_OPTIONS]
    }
    comparison_df = pd.DataFrame(comparison_data)

    # Display comparison table
    st.subheader(f"📋 Performance Metrics - {scenario_name}")
    st.dataframe(comparison_df.round(4), use_container_width=True)

    # Create 4 bar charts for each metric
    st.subheader(f"📊 Metrics Visualization - {scenario_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC-AUC Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(comparison_df["Model"], comparison_df["ROC-AUC"], 
                    color="#FF6B6B", alpha=0.8, edgecolor="black")
        ax.set_ylabel("ROC-AUC Score", fontsize=12)
        ax.set_title("ROC-AUC Comparison", fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45, ha="right")
        
        # Add value labels on bars
        for bar, value in zip(bars, comparison_df["ROC-AUC"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig, use_container_width=True)
        
        # PR-AUC Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(comparison_df["Model"], comparison_df["PR-AUC"], 
                    color="#4ECDC4", alpha=0.8, edgecolor="black")
        ax.set_ylabel("PR-AUC Score", fontsize=12)
        ax.set_title("PR-AUC Comparison", fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45, ha="right")
        
        # Add value labels on bars
        for bar, value in zip(bars, comparison_df["PR-AUC"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        # F1-Score Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(comparison_df["Model"], comparison_df["F1-Score"], 
                    color="#45B7D1", alpha=0.8, edgecolor="black")
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.set_title("F1-Score Comparison", fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45, ha="right")
        
        # Add value labels on bars
        for bar, value in zip(bars, comparison_df["F1-Score"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig, use_container_width=True)
        
        # Balanced Accuracy Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(comparison_df["Model"], comparison_df["Balanced Accuracy"], 
                    color="#96CEB4", alpha=0.8, edgecolor="black")
        ax.set_ylabel("Balanced Accuracy", fontsize=12)
        ax.set_title("Balanced Accuracy Comparison", fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45, ha="right")
        
        # Add value labels on bars
        for bar, value in zip(bars, comparison_df["Balanced Accuracy"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig, use_container_width=True)

    # Key Insights
    st.subheader("🔍 Key Insights")
    
    # Find best model for each metric
    best_roc_auc = comparison_df.loc[comparison_df["ROC-AUC"].idxmax()]
    best_pr_auc = comparison_df.loc[comparison_df["PR-AUC"].idxmax()]
    best_f1 = comparison_df.loc[comparison_df["F1-Score"].idxmax()]
    best_bal_acc = comparison_df.loc[comparison_df["Balanced Accuracy"].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Best Performing Models")
        st.write(f"**ROC-AUC:** {best_roc_auc['Model']} ({best_roc_auc['ROC-AUC']:.4f})")
        st.write(f"**PR-AUC:** {best_pr_auc['Model']} ({best_pr_auc['PR-AUC']:.4f})")
        st.write(f"**F1-Score:** {best_f1['Model']} ({best_f1['F1-Score']:.4f})")
        st.write(f"**Balanced Accuracy:** {best_bal_acc['Model']} ({best_bal_acc['Balanced Accuracy']:.4f})")
    
    with col2:
        st.markdown("### 📈 Performance Summary")
        avg_performance = comparison_df[["ROC-AUC", "PR-AUC", "F1-Score", "Balanced Accuracy"]].mean(axis=1)
        best_overall = comparison_df.loc[avg_performance.idxmax()]
        st.write(f"**Best Overall Performance:** {best_overall['Model']}")
        st.write(f"**Average Score:** {avg_performance.max():.4f}")
        
        # Performance tier classification
        max_auc = comparison_df["ROC-AUC"].max()
        if max_auc >= 0.8:
            st.success("🟢 Excellent Performance (ROC-AUC ≥ 0.8)")
        elif max_auc >= 0.7:
            st.warning("🟡 Good Performance (ROC-AUC ≥ 0.7)")
        else:
            st.error("🔴 Poor Performance (ROC-AUC < 0.7)")

    st.divider()

    st.markdown(
        """
    ### 📌 Important Takeaways
    
    1. **Duration Leakage**: The `duration` feature is only known AFTER the call, making it unusable for 
       real-world predictions. Scenario 2 reflects actual deployment conditions.
    
    2. **Time Drift**: The model shows lower performance on test data due to temporal drift. This is 
       typical in financial time series - patterns change over time.
    
    3. **Class Imbalance Handling**: SMOTE helps balance the dataset, improving recall for the minority 
       class (those who subscribe).
    
    4. **Trade-offs**:
       - Higher Recall = Catch more positive cases (but more false positives)
       - Higher Precision = Fewer false positives (but may miss some positives)
       - Use case determines which to optimize
    
    5. **Next Steps**:
       - Consider ensemble methods (Random Forest, XGBoost, LightGBM) for better non-linear patterns
       - Perform hyperparameter tuning with grid/random search
       - Monitor model performance over time and retrain periodically
    """
    )


# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 12px;">
    <p>🏦 Bank Marketing Campaign Prediction Web Demo</p>
    <p>Built with Streamlit • Machine Learning • Python</p>
    </div>
    """,
    unsafe_allow_html=True,
)