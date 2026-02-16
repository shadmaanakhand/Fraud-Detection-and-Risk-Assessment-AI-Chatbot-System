# Fraud-Detection-and-Risk-Assessment-AI-Chatbot-System

## Overview
The Fraud Detection and Risk Assessment AI Chatbot System is a machine learning-powered web application designed to detect potentially fraudulent financial transactions and provide interpretable risk explanations.

The system integrates a trained Random Forest classification model with an interactive Streamlit interface and an AI chatbot layer. It simulates a real-world fraud detection workflow used in financial institutions for transaction monitoring and decision support.

This project was developed for Artificial Intelligence (CS 4810).

---

## Features
- Real-time fraud prediction using a trained Random Forest classifier
- Probability-based fraud risk scoring
- Custom fraud threshold implementation (0.10 default)
- AI-generated explanation of model predictions using OpenAI API
- Interactive Streamlit web interface
- Secure API key management via `.env`
- Model serialization using joblib

---

## Tech Stack
- **Language:** Python  
- **Frontend / UI:** Streamlit  
- **Machine Learning:** scikit-learn (Random Forest)  
- **Data Processing:** pandas, NumPy  
- **Model Serialization:** joblib  
- **AI Integration:** OpenAI API  
- **Environment Management:** python-dotenv  
- **Visualization / Analysis:** Jupyter Notebook  

---


---

## System Architecture

### Machine Learning Model
A Random Forest classifier trained on large-scale transaction data to detect fraudulent behavior. The model outputs a fraud probability, and a custom threshold (default = 0.10) determines the final prediction.

### Inference Pipeline
1. User enters transaction details in the Streamlit UI.
2. The model computes fraud probability using `predict_proba`.
3. A custom threshold is applied.
4. The classification result is displayed.

### AI Chatbot Layer
The system sends structured transaction data and model output to the OpenAI API to generate a concise explanation of the fraud prediction. The explanation:
- Emphasizes that predictions are probabilistic
- Highlights key risk indicators
- Uses non-technical language

### User Interface
Built with Streamlit, allowing users to input:
- Transaction type
- Transaction amount
- Origin account balances
- Destination account balances

The UI displays:
- Fraud probability
- Final classification
- AI-generated explanation
- Technical debugging details

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/shadmaanakhand/Fraud-Detection-and-Risk-Assessment-AI-Chatbot-System.git
cd Fraud-Detection-and-Risk-Assessment-AI-Chatbot-System
```
### 2. Ensure Python Is Installed

Verify Python 3.9 or higher is installed:

```bash
python --version
```

If Python is not installed, download it from:

https://www.python.org/downloads/

---

### 3. Create and Activate a Virtual Environment

**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Configure Environment Variables

Create a `.env` file in the project root and add:

```
OPENAI_API_KEY=your_api_key_here
```

---

### 6. Run the Application

```bash
streamlit run fraud_detection.py
```

The application will launch locally in your browser.

------
## Future Improvements

- Implement SHAP-based model explainability
- Tune fraud threshold based on precision-recall optimization
- Add batch transaction scoring functionality
- Implement model performance monitoring
- Containerize with Docker
- Deploy to cloud infrastructure
