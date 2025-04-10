import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt


def neural_network_section():
    st.header("🧠 Neural Network Classifier")
    st.write("Upload your classification dataset, select the target column, and train a neural network.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="nn_file")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        columns = data.columns.tolist()
        target_column = st.selectbox("Select the target column", columns)
        feature_columns = st.multiselect("Select feature columns", [col for col in columns if col != target_column])

        if not feature_columns:
            st.warning("Please select at least one feature column.")
            return

        if st.checkbox("Drop rows with missing values"):
            data = data.dropna(subset=feature_columns + [target_column])
            st.success("Dropped rows with missing values.")

        X = data[feature_columns].values
        y = data[target_column].values

        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)
            label_names = le.classes_
        else:
            label_names = np.unique(y)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        st.subheader("Hyperparameters")
        epochs = st.slider("Epochs", 5, 100, 20)
        batch_size = st.slider("Batch size", 8, 128, 32)
        learning_rate = st.number_input("Learning rate", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / epochs) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f} - Acc: {logs['accuracy']:.4f}")

        with st.spinner("Training neural network..."):
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[StreamlitCallback()]
            )

        st.subheader("Training Metrics")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Val Loss')
        ax1.set_title("Loss")
        ax1.legend()

        ax2.plot(history.history['accuracy'], label='Train Acc')
        ax2.plot(history.history['val_accuracy'], label='Val Acc')
        ax2.set_title("Accuracy")
        ax2.legend()
        st.pyplot(fig)

        st.subheader("Make a Prediction")
        user_input = {}
        for feature in feature_columns:
            user_input[feature] = st.number_input(f"Enter value for {feature}", value=float(data[feature].mean()))

        if st.button("Predict"):
            user_df = pd.DataFrame([user_input])
            scaled_input = scaler.transform(user_df.values)
            prediction = model.predict(scaled_input)
            pred_class = np.argmax(prediction, axis=1)[0]
            if 'le' in locals():
                st.success(f"Predicted Class: {le.inverse_transform([pred_class])[0]}")
            else:
                st.success(f"Predicted Class Index: {pred_class}")
