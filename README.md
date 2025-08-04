Privilege Escalation Attack Detection and Mitigation in Cloud using Machine Learning

 Overview

This project aims to enhance the security of cloud environments by detecting and mitigating **privilege escalation attacks** using machine learning techniques. Privilege escalation is a critical threat where attackers gain unauthorized elevated access to resources. Our system intelligently identifies suspicious behavior and automatically responds to potential threats in real time.

 Objectives

- Detect unauthorized privilege escalation attempts in the cloud.
- Automate mitigation actions upon detection.
- Ensure minimal false positives while maintaining high detection accuracy.

 Tech Stack

- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, CatBoost, Matplotlib, Seaborn
- **Preprocessing**: Standard Scaler (for data normalization)
- **Model**: CatBoost Classifier
- **Environment**: Jupyter Notebook / Google Colab

 Machine Learning Workflow

1. **Data Collection**: Cloud activity logs simulating user behavior.
2. **Data Preprocessing**: Null handling, normalization with Standard Scaler.
3. **Feature Engineering**: Behavioral pattern extraction.
4. **Model Training**: CatBoost used for classification due to its strength with categorical data and handling imbalanced classes.
5. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score.
6. **Mitigation**: Automated triggers for revoking access or alerting admins.

 Project Structure

```bash
├── data/                      # Input datasets (cloud logs)
├── notebooks/                 # Jupyter notebooks for each phase
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
├── src/                       # Core scripts for model and pipeline
│   ├── model.py
│   ├── detection.py
│   └── mitigation.py
├── results/                   # Model performance reports and charts
├── README.md                  # Project documentation
└── requirements.txt           # Dependencies
