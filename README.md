# Intrusion-Detection-System
# 🧠 Intrusion Detection System using NSL-KDD Dataset and SVM

This project implements a **Network Intrusion Detection System (IDS)** using the **NSL-KDD dataset** and a **Support Vector Machine (SVM)** model.  
It analyzes network connection records to classify them as *normal* or *attack*, achieving high accuracy through effective preprocessing and model tuning.

---

## 📘 Project Overview

Network intrusion detection is a critical task in cybersecurity, identifying malicious activities within network traffic.  
This notebook builds an IDS model using the NSL-KDD dataset and evaluates it using multiple SVM kernel functions (linear, polynomial, RBF).

---

## ⚙️ Features

- Data preprocessing and feature encoding
- Training and testing on NSL-KDD dataset
- Evaluation of SVM classifiers with different kernels
- Accuracy, confusion matrix, and classification report
- Visualizations for model performance comparison

---

## 🧾 Dataset

**Dataset:** [NSL-KDD Dataset on Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd)

### Description:
The NSL-KDD dataset is a refined version of the classic KDD Cup 1999 dataset, designed to benchmark intrusion detection systems by removing redundant records and balancing class distribution.

### Dataset includes:
- **41 features** describing each network connection  
- **Labels:** `normal` or `attack` (DoS, Probe, R2L, U2R)

---

## 🧰 Technologies Used

| Category | Library/Tool |
|-----------|---------------|
| Language | Python 3.10+ |
| IDE | Jupyter Notebook / VS Code |
| Data Handling | pandas, numpy |
| Machine Learning | scikit-learn (SVC) |
| Visualization | matplotlib, seaborn |
| Dataset Source | KaggleHub / Manual download |

---

## 🧪 Model Workflow

1. **Load Dataset**
   - Downloaded from Kaggle via `kagglehub` or manual CSV import.
2. **Preprocessing**
   - Encode categorical features (`protocol_type`, `service`, `flag`) using `LabelEncoder`
   - Normalize numerical features using `StandardScaler`
3. **Model Training**
   - Tested multiple SVM kernels: `linear`, `poly`, `rbf`
   - Selected best model based on accuracy and generalization
4. **Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

---

## 📊 Results

| Kernel | Accuracy |
|:--------|:----------|
| Linear | ~97% |
| Polynomial | ~98% |
| **RBF (Best)** | **99.5%** |

The **RBF kernel SVM** achieved the highest performance with strong detection of both normal and attack traffic classes.

---

## 📈 Visualizations

The notebook includes:
- Accuracy comparison plots for different kernels  
- Confusion matrix heatmaps  
- Classification metrics summary  

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/2024yuva/Intrusion-Detection-System.git
cd intrusion-detection-system
```

###  Create a virtual Environment
```
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
3️⃣ Install dependencies

```
pip install -r requirements.txt
```

4️⃣ Run the notebook
```
Open the notebook in VS Code or Jupyter:
```
---

Jupyter Notebook "Intrusion Detection System NSL_KDD+SVM.ipynb"


Run all cells to train, test, and evaluate the model.

📂 Project Structure
```
IDS-NSL-KDD/
│
├── Intrusion Detection System NSL_KDD+SVM.ipynb    # Main notebook
├── data/                                           # Dataset folder
├── models/                                         # Saved model files (optional)
├── requirements.txt                                # Python dependencies
└── README.md                                       # Project documentation

```

### Future Enchancement
Extend model using Deep Learning (CNN, LSTM)

Deploy on cloud (Render / Google Cloud Run)

👩‍💻 Author

Yuvarrunjitha R A
📧 yuvarrunjithars@gmail.com

💻 GitHub: @2024yuva

🪪 License

This project is licensed under the MIT License.
Feel free to use, modify, and share for educational and research purposes.

