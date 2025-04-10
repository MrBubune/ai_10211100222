# 🧠 ML & AI Explorer

Welcome to the **ML & AI Explorer** — an interactive Streamlit application designed to help users explore and solve various Machine Learning and AI problems, including Regression, Clustering, Neural Networks, and Large Language Models (LLMs). This tool offers separate sections for each task, complete with interactive interfaces, visualizations, and prediction capabilities.

![ML & AI Explorer Screenshot](images/ml_ai_explorer_screenshot.png)

## 🚀 Features

- **Regression**: Build linear models to predict continuous variables with dataset upload, preprocessing options, performance metrics, and custom predictions.
  
- **Clustering**: Perform K-Means clustering with interactive selection of features and number of clusters, visualized in 2D or 3D plots.

- **Neural Networks**: Train feedforward neural networks for classification tasks with real-time training progress, hyperparameter tuning, and custom predictions.

- **LLM Q&A**: Engage with a Large Language Model using text, documents, or images to perform question-answering tasks.

## 🏗️ Architecture

The application is structured into modular sections, each handling a specific ML or AI task. The navigation is managed via Streamlit's sidebar, allowing seamless transitions between different functionalities.

![Application Architecture](images/application_architecture.png)

## 📂 Directory Structure

```
ML_AI_Explorer/
├── app.py
├── sections/
│   ├── __init__.py
│   ├── home.py
│   ├── regression.py
│   ├── clustering.py
│   ├── neural_network.py
│   └── llm_multimodal.py
├── requirements.txt
└── README.md
```

- `app.py`: The main entry point that sets up the Streamlit interface and navigation.
- `sections/`: Contains individual modules for each section of the application.
- `requirements.txt`: Lists all necessary Python packages with specific versions.
- `README.md`: This documentation file.

## 🛠️ Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/ML_AI_Explorer.git
   cd ML_AI_Explorer
   ```

2. **Set Up a Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  
   On Windows, use 'venv\Scripts\activate'
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:

   Create a `.env` file in the root directory and add your API keys:

   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

## ▶️ Usage

Run the application with:

```cmd
streamlit run app.py
```

Navigate through the sections using the sidebar to explore different ML and AI functionalities.

## 🧩 Sections Overview

### Home

Provides an overview of the application and its features.

![Home Section Screenshot](images/home_section_screenshot.png)

### Regression

- **Functionality**: Upload datasets, select target and feature columns, preprocess data, train a linear regression model, evaluate performance, and make custom predictions.
  
- **Visualizations**: Scatter plot of actual vs. predicted values with regression line.

![Regression Section Screenshot](images/regression_section_screenshot.png)

### Clustering

- **Functionality**: Upload datasets, select features, choose the number of clusters, apply K-Means clustering, and download clustered data.
  
- **Visualizations**: Interactive 2D and 3D scatter plots of clustered data using Plotly.

![Clustering Section Screenshot](images/clustering_section_screenshot.png)

### Neural Network

- **Functionality**: Upload datasets for classification, select target and feature columns, preprocess data, configure hyperparameters, train a feedforward neural network, and make custom predictions.
  
- **Visualizations**: Real-time training progress with loss and accuracy plots.

![Neural Network Section Screenshot](images/neural_network_section_screenshot.png)

### LLM Q&A

- **Functionality**: Interact with a Large Language Model using text inputs, documents (PDF, TXT, CSV), or images to perform question-answering tasks.
  
- **Visualizations**: Display of extracted content and generated responses.

![LLM Q&A Section Screenshot](images/llm_qa_section_screenshot.png)

## 🖼️ LLM Architecture Diagram

The LLM Q&A section utilizes the following architecture:

![LLM Architecture Diagram](images/llm_architecture_diagram.png)

## 📝 Methodology

1. **Data Input**: Users provide input in the form of text, documents, or images.
2. **Preprocessing**: The input is processed to extract relevant content.
3. **Model Interaction**: The processed content is sent to the Gemini AI API.
4. **Response Generation**: The model generates a response based on the input.
5. **Output Display**: The response is displayed to the user within the Streamlit interface.

## 📊 Comparison with ChatGPT

| Feature               | ML & AI Explorer (Gemini AI) | ChatGPT                     |
|-----------------------|------------------------------|-----------------------------|
| **Multimodal Input**  | Supports text, documents, and images | Primarily text-based        |
| **Real-time Interaction** | Yes                      | Yes                         |
| **Customization**     | User-defined input and preprocessing | Limited to text prompts     |
| **Integration**       | Seamless integration with Streamlit | Requires additional setup   |

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

## 📜 License

This project is licensed under the MIT License.

---