import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

def neural_network_section():
    st.header("🧠 Neural Network Classifier")
    st.write("Upload a classification dataset, configure the neural network, and view predictions and performance.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="neural_network")
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

        # Preprocessing
        if st.checkbox("Drop rows with missing values"):
            data = data.dropna(subset=feature_columns + [target_column])
            st.success("Missing rows dropped.")

        # Classification check
        if data[target_column].nunique() > 50 and not st.checkbox("Convert continuous target to classes (binning)"):
            st.error("Target column seems continuous (regression). Neural networks here support only classification.")
            return

        # Optional binning
        if data[target_column].nunique() > 50 and st.checkbox("Convert target to 10 classes (binned)"):
            data[target_column] = pd.qcut(data[target_column], q=10, labels=False)
            st.info("Target column binned into 10 categories.")

        X = data[feature_columns].values
        y = data[target_column].values

        # Encode if labels are not numeric
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            le = None

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        test_size = st.slider("Test size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        st.subheader("Model Hyperparameters")
        epochs = st.slider("Epochs", 1, 100, 10)
        batch_size = st.slider("Batch size", 8, 128, 32)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001, format="%.4f")

        num_classes = len(np.unique(y_train))
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        st.subheader("📈 Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()

        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{epochs}")

        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[StreamlitCallback()]
            )
        except Exception as e:
            st.error(f"Error during training: {e}")
            return

        # Plot metrics
        st.subheader("📊 Training Metrics")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Val Loss')
        ax1.set_title("Loss")
        ax1.legend()

        ax2.plot(history.history['accuracy'], label='Train Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax2.set_title("Accuracy")
        ax2.legend()
        st.pyplot(fig)

        st.success(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")

        # Custom prediction
        st.subheader("🔍 Custom Prediction")
        custom_input = {}
        for feature in feature_columns:
            default_val = float(data[feature].mean())
            custom_input[feature] = st.number_input(f"Enter value for {feature}", value=default_val)

        if st.button("Predict"):
            custom_df = pd.DataFrame([custom_input])
            custom_scaled = scaler.transform(custom_df)
            pred = model.predict(custom_scaled)
            pred_class = np.argmax(pred, axis=1)[0]
            class_name = le.inverse_transform([pred_class])[0] if le else pred_class
            st.success(f"Predicted {target_column}: {class_name}")
    else:
        st.info("Please upload a classification dataset to get started.")
