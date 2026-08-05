# ⚡ Electricity Consumption Prediction Using XGBoost & Flask

An AI-powered **Electricity Consumption Prediction Web Application** that predicts electricity usage using **XGBoost Regression** through an interactive **Flask Web Interface**.

The application enables users to enter input parameters and instantly predict electricity consumption using a trained machine learning model.

---

# 🌐 Web Application

The web application allows users to:

- ⚡ Enter electricity-related input parameters
- 📊 Predict electricity consumption instantly
- 📈 View prediction results in real time
- 💻 Use a responsive and user-friendly interface

---

# 📌 Project Overview

Electricity consumption forecasting plays a vital role in energy management, smart grids, and resource planning.

This project uses the **XGBoost Regressor** to analyze historical electricity consumption data and predict future electricity usage accurately. The trained model is integrated into a **Flask web application**, allowing users to generate predictions through an intuitive browser interface.

---

# 🚀 Features

- ✅ Data Cleaning & Preprocessing
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature Engineering
- ✅ XGBoost Regression Model
- ✅ Model Evaluation
- ✅ Saved Trained Model
- ✅ Flask Web Application
- ✅ Responsive HTML Interface
- ✅ Real-time Electricity Consumption Prediction
- ✅ Easy Deployment

---

# 📂 Project Structure

```text
electricity-consumption-prediction-using-xgboost/

│
├── dataset/
│   └── electricity_consumption.csv
│
├── models/
│   └── electricity_xgboost_model.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── app.py
├── electricity_prediction.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The dataset contains historical electricity consumption records and related features used to train the prediction model.

### Input Features

- Historical Electricity Usage
- Time-based Features
- Environmental Factors (if applicable)
- Additional Energy Consumption Parameters

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Machine Learning

- XGBoost Regression
- Scikit-learn

## Web Development

- Flask
- HTML5
- CSS3
- JavaScript

## Libraries

- Pandas
- NumPy
- Matplotlib
- Joblib

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/khanzainabshuaib/electricity-consumption-prediction-using-xgboost.git
```

Navigate to the project folder

```bash
cd electricity-consumption-prediction-using-xgboost
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Flask application

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

Enter the required input values and click **Predict** to generate the electricity consumption prediction.

---

# 📈 Model Performance

## Algorithm

XGBoost Regression

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

The model provides accurate electricity consumption predictions using historical energy data.

---

# 📸 Application Screenshots

## Home Page

*(Add a screenshot here)*

## Prediction Result

<img width="940" height="474" alt="image" src="https://github.com/user-attachments/assets/88340e5d-b86b-473d-a3c2-2cf13d51a9d7" />


---

# 💡 Example Prediction

### Input

- Historical Consumption: 310 kWh
- Time Period: Evening
- Additional Parameters: Sample Values

### Output

```text
Predicted Electricity Consumption:
325.67 kWh
```

---

# 🔮 Future Improvements

- Live Electricity Data Integration
- Smart Meter API Support
- Interactive Dashboard
- FastAPI Version
- Cloud Deployment (AWS, Azure, Render)
- Mobile-Friendly Interface
- Time-Series Forecasting with LSTM

---

# 📚 Learning Outcomes

- XGBoost Regression
- Data Preprocessing
- Feature Engineering
- Model Evaluation
- Flask Web Development
- Machine Learning Deployment
- End-to-End ML Pipeline

---

# 👨‍💻 Author

**Zainab Khan**

Machine Learning | Artificial Intelligence | Python Developer

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐.

Contributions, suggestions, and feedback are always welcome!
