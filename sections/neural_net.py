import streamlit as st
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def show():
    st.header("🧠 Neural Network Task")

    uploaded_file = st.file_uploader("Upload CSV Dataset for Classification", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview", df.head())
        
        target_col = st.text_input("Target Column Name")

        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            X = StandardScaler().fit_transform(X)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

            epochs = st.slider("Epochs", 5, 100, 10)
            lr = st.slider("Learning Rate", 0.001, 0.01, 0.001)

            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(len(set(y)), activation='softmax')
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=0)

            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(history.history['loss'], label='Train Loss')
            ax[0].plot(history.history['val_loss'], label='Val Loss')
            ax[0].legend()
            ax[1].plot(history.history['accuracy'], label='Train Acc')
            ax[1].plot(history.history['val_accuracy'], label='Val Acc')
            ax[1].legend()
            st.pyplot(fig)

            st.subheader("Prediction")
            test_sample = {}
            for i, col in enumerate(df.drop(columns=[target_col]).columns):
                test_sample[col] = st.number_input(f"{col}", value=0.0)
            input_df = pd.DataFrame([test_sample])
            input_scaled = StandardScaler().fit_transform(input_df)
            pred = model.predict(input_scaled)
            st.success(f"Predicted Class: {pred.argmax()}")
