import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def regression_section():
    st.header("📈 Regression Analysis")
    st.write("Upload your regression dataset, select the target column, and view predictions and visualizations.")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        if st.checkbox("Show dataset summary"):
            st.write(data.describe())

        columns = data.columns.tolist()
        target_column = st.selectbox("Select target column", columns)
        feature_columns = st.multiselect("Select feature columns", [col for col in columns if col != target_column])

        if not feature_columns:
            st.warning("Please select at least one feature column.")
            return

        if st.checkbox("Drop rows with missing values"):
            data = data.dropna(subset=feature_columns + [target_column])
            st.success("Dropped rows with missing values.")

        X = data[feature_columns]
        y = data[target_column]

        if not all(np.issubdtype(X[feat].dtype, np.number) for feat in feature_columns):
            st.error("All selected features must be numeric for regression.")
            return

        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Performance")
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        st.metric("R² Score", f"{r2:.2f}")

        st.subheader("Actual vs Predicted Values")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        if len(feature_columns) == 1:
            st.subheader("Regression Line Visualization")
            fig2, ax2 = plt.subplots()
            ax2.scatter(X_test, y_test, label="Actual")
            ax2.plot(X_test, y_pred, color='red', label="Regression Line")
            ax2.set_xlabel(feature_columns[0])
            ax2.set_ylabel(target_column)
            ax2.set_title("Regression Line")
            ax2.legend()
            st.pyplot(fig2)

        st.subheader("Custom Prediction")
        user_input = {}
        for feature in feature_columns:
            user_input[feature] = st.number_input(f"Enter value for {feature}", value=float(X[feature].mean()))

        if st.button("Predict"):
            user_df = pd.DataFrame([user_input])
            prediction = model.predict(user_df)[0]
            st.success(f"Predicted {target_column}: {prediction:.2f}")