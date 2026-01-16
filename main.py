import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris, load_breast_cancer
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Triangle ML",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_datasets():
    """Load sample datasets for demonstration"""
    datasets = {
        "Iris": load_iris(),
        "Breast Cancer": load_breast_cancer(),
        "Synthetic Binary": make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42),
        "Synthetic Multi-class": make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=3, n_redundant=2, random_state=42)
    }
    return datasets

def main():
    st.markdown('<h1 class="main-header">🔺 Triangle ML</h1>', unsafe_allow_html=True)
    st.markdown("### Machine Learning Experimentation Platform")

    # Sidebar for controls
    with st.sidebar:
        st.markdown('<h3 class="sidebar-header">⚙️ Configuration</h3>', unsafe_allow_html=True)

        # Dataset selection
        st.subheader("📊 Dataset Selection")
        dataset_option = st.selectbox(
            "Choose a dataset:",
            ["Iris", "Breast Cancer", "Synthetic Binary", "Synthetic Multi-class", "Upload Custom"]
        )

        # Algorithm selection
        st.subheader("🤖 Algorithm Selection")
        algorithm = st.selectbox(
            "Choose algorithm:",
            ["Random Forest", "SVM", "Logistic Regression", "XGBoost", "LightGBM"]
        )

        # Hyperparameters section
        st.subheader("🔧 Hyperparameters")

        if algorithm == "Random Forest":
            n_estimators = st.slider("Number of trees", 10, 500, 100)
            max_depth = st.slider("Max depth", 1, 50, 10)
            min_samples_split = st.slider("Min samples split", 2, 20, 2)
            hyperparameters = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split
            }

        elif algorithm == "SVM":
            C = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
            gamma = st.selectbox("Gamma", ["scale", "auto"]) if kernel != "linear" else "scale"
            hyperparameters = {
                "C": C,
                "kernel": kernel,
                "gamma": gamma
            }

        elif algorithm == "Logistic Regression":
            C = st.slider("C (Inverse regularization)", 0.01, 10.0, 1.0)
            penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"])
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"])
            hyperparameters = {
                "C": C,
                "penalty": penalty,
                "solver": solver
            }

        elif algorithm in ["XGBoost", "LightGBM"]:
            n_estimators = st.slider("Number of estimators", 10, 500, 100)
            learning_rate = st.slider("Learning rate", 0.01, 1.0, 0.1)
            max_depth = st.slider("Max depth", 1, 20, 6)
            hyperparameters = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth
            }

        # Training parameters
        st.subheader("🎯 Training Parameters")
        test_size = st.slider("Test size (%)", 10, 50, 20)
        random_state = st.number_input("Random state", value=42, min_value=0)

        # Run button
        run_experiment = st.button("🚀 Run Experiment", type="primary")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Results & Visualizations")

        if run_experiment:
            with st.spinner("Training model..."):
                # Load dataset
                datasets = load_sample_datasets()
                if dataset_option in datasets:
                    data = datasets[dataset_option]
                    X, y = data.data, data.target
                    feature_names = data.feature_names if hasattr(data, 'feature_names') else [f'feature_{i}' for i in range(X.shape[1])]
                    target_names = data.target_names if hasattr(data, 'target_names') else None
                else:
                    st.error("Custom dataset upload not implemented yet")
                    return

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=random_state
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train model
                if algorithm == "Random Forest":
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(**hyperparameters, random_state=random_state)
                elif algorithm == "SVM":
                    from sklearn.svm import SVC
                    model = SVC(**hyperparameters, random_state=random_state)
                elif algorithm == "Logistic Regression":
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(**hyperparameters, random_state=random_state, max_iter=1000)
                elif algorithm == "XGBoost":
                    from xgboost import XGBClassifier
                    model = XGBClassifier(**hyperparameters, random_state=random_state)
                elif algorithm == "LightGBM":
                    from lightgbm import LGBMClassifier
                    model = LGBMClassifier(**hyperparameters, random_state=random_state, verbosity=-1)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)

                # Display results
                st.success("✅ Model trained successfully!")

                # Metrics
                st.subheader("📊 Model Performance")
                metric_col1, metric_col2, metric_col3 = st.columns(3)

                with metric_col1:
                    st.metric("Accuracy", f"{accuracy:.3f}")

                with metric_col2:
                    st.metric("Training Samples", len(X_train))

                with metric_col3:
                    st.metric("Test Samples", len(X_test))

                # Classification report
                if len(np.unique(y)) <= 10:  # Only show for reasonable number of classes
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0))

                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("🎯 Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                               title='Feature Importance')
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👈 Configure your experiment in the sidebar and click 'Run Experiment' to start!")

    with col2:
        st.subheader("📋 Dataset Info")

        if run_experiment:
            datasets = load_sample_datasets()
            if dataset_option in datasets:
                data = datasets[dataset_option]
                st.write(f"**Dataset:** {dataset_option}")
                st.write(f"**Samples:** {data.data.shape[0]}")
                st.write(f"**Features:** {data.data.shape[1]}")
                st.write(f"**Classes:** {len(np.unique(data.target))}")

                # Show target distribution
                if len(np.unique(data.target)) <= 10:
                    target_counts = pd.Series(data.target).value_counts()
                    fig = px.pie(values=target_counts.values, names=target_counts.index,
                               title="Target Distribution")
                    st.plotly_chart(fig, use_container_width=True)

        # Hyperparameter summary
        if run_experiment:
            st.subheader("🔧 Current Hyperparameters")
            for param, value in hyperparameters.items():
                st.write(f"**{param}:** {value}")

if __name__ == "__main__":
    main()