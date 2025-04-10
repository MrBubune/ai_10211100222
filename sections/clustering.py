import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import io

def clustering_section():
    st.header("🔍 Clustering Explorer")
    st.write("Upload your dataset, choose features, and explore clusters interactively.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="clustering")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        columns = data.columns.tolist()
        selected_features = st.multiselect("Select features for clustering", columns)
        if len(selected_features) < 2:
            st.warning("Please select at least 2 features.")
            return

        if st.checkbox("Drop rows with missing values"):
            data = data.dropna(subset=selected_features)
            st.success("Dropped rows with missing values.")

        X = data[selected_features]

        if not all(np.issubdtype(X[feat].dtype, np.number) for feat in selected_features):
            st.error("All selected features must be numeric.")
            return

        num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X)
        data['Cluster'] = clusters

        st.subheader("Cluster Summary")
        st.dataframe(data.groupby('Cluster').mean().reset_index())

        st.subheader("Interactive Cluster Visualization")
        if len(selected_features) == 2:
            fig = px.scatter(
                data, x=selected_features[0], y=selected_features[1],
                color=data['Cluster'].astype(str), symbol=data['Cluster'].astype(str),
                title="2D Cluster Visualization", labels={"color": "Cluster"}
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig, use_container_width=True)

        elif len(selected_features) == 3:
            fig = px.scatter_3d(
                data, x=selected_features[0], y=selected_features[1], z=selected_features[2],
                color=data['Cluster'].astype(str), title="3D Cluster Visualization"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            pca = PCA(n_components=3)
            components = pca.fit_transform(X)
            data[['PC1', 'PC2', 'PC3']] = components
            fig = px.scatter_3d(
                data, x='PC1', y='PC2', z='PC3', color=data['Cluster'].astype(str),
                title="PCA-based 3D Clustering Visualization"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Download Clustered Dataset")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV with Clusters",
            data=csv,
            file_name="clustered_output.csv",
            mime="text/csv"
        )