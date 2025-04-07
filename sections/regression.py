import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def show():
    st.header("📈 Regression Task")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview Dataset", df.head())

        target_col = st.text_input("Enter the name of the target column")

        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            st.write("Feature Columns:", list(X.columns))

            if st.checkbox("Train Model"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                st.subheader("Performance Metrics")
                st.write("MAE:", mean_absolute_error(y_test, predictions))
                st.write("R² Score:", r2_score(y_test, predictions))

                fig, ax = plt.subplots()
                ax.scatter(y_test, predictions)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)

                st.subheader("Make Prediction")
                custom_input = {}
                for col in X.columns:
                    custom_input[col] = st.number_input(f"{col}", value=0.0)
                input_df = pd.DataFrame([custom_input])
                pred = model.predict(input_df)
                st.success(f"Predicted Value: {pred[0]}")
