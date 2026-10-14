# Employee Churn Prediction using Machine Learning

## 📌 Project Overview

Employee Churn Prediction is a Machine Learning project that predicts whether an employee is likely to leave a company based on their personal and professional details.

The project uses the **Gaussian Naive Bayes algorithm** to make predictions through a web application built with Flask. Users can enter employee information and get a prediction from the trained model.

The main goal of this project is to demonstrate how Machine Learning can be used to understand employee attrition and support HR decision-making.

## 🎯 Project Objectives

- Predict employee churn using Machine Learning.
- Build an interactive web application using Flask.
- Use a trained model to generate predictions.
- Accept employee details through a user-friendly interface.
- Understand how Machine Learning can be applied to HR analytics.

## ✨ Features

- Employee churn prediction using a trained ML model.
- Interactive web interface built with Flask.
- Input validation for employee information.
- Prediction results displayed on the web page.
- Prediction probabilities when supported by the model.
- Light and dark mode interface.
- Reusable trained model saved as a `.pkl` file.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **ML Algorithm:** Gaussian Naive Bayes
- **Web Framework:** Flask
- **Data Handling:** Pandas
- **Model Loading:** Joblib
- **Deployment Server:** Gunicorn

## 📋 Input Features

The application uses the following employee details for prediction:

| Feature | Description |
|---|---|
| Education | Employee's education level |
| Joining Year | Year the employee joined the company |
| City | Employee's city |
| Payment Tier | Employee's payment category |
| Age | Employee's age |
| Gender | Employee's gender |
| Ever Benched | Whether the employee has previously been benched |
| Experience in Current Domain | Years of experience in the current domain |

## ⚙️ Project Structure

```text
Employee_Churn_Prediction/
│
├── app.py              # Flask web application
├── naive_bayes.pkl     # Trained Naive Bayes model
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🚀 Installation and Setup

Follow these steps to run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/ashutosh-dikkatwar/Employee_Churn_Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Employee_Churn_Prediction
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

### 6. Open the Application

Open your browser and visit:

```text
http://127.0.0.1:5000
```

Enter the employee details in the form and submit them to view the prediction.

## 🧠 Machine Learning Workflow

1. **Data Preparation:** Prepare employee information in the format expected by the model.
2. **Model Training:** Train a Gaussian Naive Bayes classifier using employee data.
3. **Model Saving:** Save the trained model as `naive_bayes.pkl`.
4. **Model Loading:** Load the saved model into the Flask application using Joblib.
5. **User Input:** Collect employee details through the web interface.
6. **Prediction:** Pass the input features to the trained model.
7. **Result Display:** Display the prediction on the web page.

## 💡 Use Cases

- Employee attrition analysis.
- HR analytics learning and experimentation.
- Understanding how classification models can support employee retention planning.
- Demonstrating an end-to-end Machine Learning application.

## 📈 Future Improvements

- Add model evaluation metrics such as accuracy, precision, recall, and F1-score.
- Include visualizations for employee attrition patterns.
- Improve prediction explanations and feature insights.
- Add more detailed reports for HR analysis.
- Deploy the application to a cloud platform.

## 👨‍💻 Author

**Ashutosh Dikkatwar**

- GitHub: [ashutosh-dikkatwar](https://github.com/ashutosh-dikkatwar)
- Project Repository: [Employee Churn Prediction](https://github.com/ashutosh-dikkatwar/Employee_Churn_Prediction)

---

⭐ If you find this project useful, consider giving the repository a star!
