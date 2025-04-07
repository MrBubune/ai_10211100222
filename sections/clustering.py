import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show():
    st.header("🔗 Clustering Task")

    uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview Dataset", df.head())
        
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        k = st.slider("Select Number of Clusters", 2, 10, 3)

        if st.button("Run K-Means Clustering"):
            model = KMeans(n_clusters=k)
            df['Cluster'] = model.fit_predict(numeric_df)

            if numeric_df.shape[1] >= 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(numeric_df.iloc[:,0], numeric_df.iloc[:,1], numeric_df.iloc[:,2], c=df['Cluster'], cmap='viridis')
                ax.set_xlabel(numeric_df.columns[0])
                ax.set_ylabel(numeric_df.columns[1])
                ax.set_zlabel(numeric_df.columns[2]) # type: ignore
                st.pyplot(fig)
            else:
                st.write("📉 2D Cluster Plot")
                plt.scatter(numeric_df.iloc[:, 0], numeric_df.iloc[:, 1], c=df['Cluster'], cmap='viridis')
                plt.xlabel(numeric_df.columns[0])
                plt.ylabel(numeric_df.columns[1])
                st.pyplot(plt)

            st.download_button("Download Clustered Data", df.to_csv(index=False), "clustered_data.csv")
