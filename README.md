🎥 Predict Audience Ratings with Machine Learning

This project predicts audience ratings for movies using machine learning. The pipeline includes data preprocessing, model training, evaluation, and saving the results. The dataset used is "Rotten Tomatoes Movies."

🔍 Features

Load data from an Excel file.

Preprocess the dataset by handling missing values and encoding categorical data.

Build and train a random forest regression model.

Evaluate the model using metrics like Mean Squared Error (MSE) and R2 Score.

Save the trained model and scaler for future use.

Generate predictions for all rows in the dataset and save the results.

🛠️ Tools and Libraries Used

Python

pandas, numpy

scikit-learn

matplotlib, seaborn

joblib

📋 How to Use

Place the dataset in the specified path.

Run the script to:

Preprocess the data.

Train the random forest regression model.

Evaluate the model's performance.

Save predictions and the trained model.

Check the output files for predictions and model artifacts.

✅ Validation

The model is validated using:

Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.

R2 Score: Indicates how well the model explains the variance in the target variable.

🌟 Advantages of Random Forest over Linear Regression

Non-Linearity: Random Forest can model complex relationships between features and the target variable, unlike linear regression which assumes a linear relationship.

Feature Importance: Random Forest provides insights into feature importance, helping identify the most significant variables.

Robustness to Outliers: Random Forest is less sensitive to outliers in the data compared to linear regression.

Handles High-Dimensional Data: Performs well even when there are many input features.

Automatic Handling of Interactions: Random Forest captures interactions between variables without explicitly specifying them.

This project demonstrates building a machine learning pipeline using Random Forest to predict audience ratings efficiently.
The differnce can be find using this
Calculate the percentage improvement using:
Improvement (%)=(Metric Random Forest − Metric Linear Regression)/Metric Linear Regression × 100

​

The output 
![image](https://github.com/user-attachments/assets/d68491fd-a23d-4098-aa0b-d3125c34a98b)
