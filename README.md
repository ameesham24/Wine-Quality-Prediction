This project uses machine learning and interactive visualization to predict the quality of red wine based on physicochemical properties. It includes a complete ML pipeline and a modern, interactive web application built with Streamlit.

🔍 Dataset: Wine Quality (Red) from UCI / Kaggle
🧠 ML Models: Logistic Regression, Random Forest, SVM
🚀 Deployment: Streamlit web app with real-time predictions
wine_quality_prediction/
│
├── data/
│   └── WineQT.csv                  # Dataset
│
├── models/
│   └── best_model.pkl             # Trained ML model (joblib/pickle)
│
├── outputs/
│   ├── test_metrics.csv           # Evaluation metrics (accuracy, precision, etc.)
│   ├── X_test.csv                 # Test feature set
│   └── y_test.csv                 # Test labels
│
├── app.py                         # Streamlit application
├── model_training.ipynb          # ML model development (Jupyter Notebook)
├── requirements.txt              # Required Python packages
└── README.md                      # Project overview (this file)
📦 Features
✅ End-to-End ML Pipeline
✅ Interactive Visualizations (Plotly)
✅ Real-Time Predictions
✅ Evaluation Metrics + Confusion Matrix
✅ Clean UI with Sidebar Navigation
✅ Responsive & Styled using Streamlit widgets
🧠 Model Training
The training process includes:

Data loading and preprocessing

Feature engineering (labeling good quality wines)

Splitting into train/test sets

Training multiple models (Logistic Regression, Random Forest, SVM)

Model selection using cross-validation

Saving the best model using joblib

🔧 Training script: model_training.ipynb
🌐 Streamlit App Overview
Launch the app:
streamlit run app.py
🧭 App Pages:

📊 Data Exploration

Dataset shape, column info, sample rows

Missing value check

Interactive filtering by wine quality

📈 Visualisations

Quality distribution (histogram)

Correlation heatmap

Alcohol vs Quality (scatter)

Feature-specific histograms

🤖 Prediction

User inputs for wine features

Real-time quality prediction

Confidence score (if available)

📋 Model Performance

Accuracy, precision, recall, F1

Confusion matrix heatmap

Model comparison metrics

⚙️ Installation & Setup
Clone this repo or download as ZIP

Create & activate virtual environment:

Windows:python -m venv venv
venv\Scripts\activate
Mac/Linux:python3 -m venv venv
source venv/bin/activate
Install requirements:pip install -r requirements.txt
Run the app:streamlit run app.py
📊 Dataset Info
Name: Wine Quality Dataset

Type: Multivariate (Classification)

Source: UCI / Kaggle

Rows: ~1,500

Features: 11 numeric + 1 quality score (0–10)

Key Columns:

fixed acidity, volatile acidity, citric acid, residual sugar, chlorides

free sulfur dioxide, total sulfur dioxide, density, pH, sulphates

alcohol, quality (target), Id

Target variable:

quality_label = 1 (good wine, quality ≥ 7)

quality_label = 0 (not good wine, quality < 7)

📈 Evaluation Metrics
Saved to outputs/test_metrics.csv and displayed in the app.

Includes:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

🛠️ Technologies Used
Python 3.10+

Streamlit

pandas, numpy

scikit-learn

plotly, matplotlib, seaborn

joblib

Jupyter Notebook (for training)

✅ To Do / Improvements
Add white wine dataset (multi-dataset comparison)

Hyperparameter tuning via GridSearchCV

Model explainability (e.g., SHAP)

Docker deployment or cloud hosting (e.g., Streamlit Cloud)

👤 Author
🧑 ameesham24
📧 ameeshamalinda@gmail.com
📁 GitHub: github.com/yourusername

📜 License
This project is licensed under the MIT License.


