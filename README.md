# END-TO-END-DATA-SCIENCE-PROJECT

COMPANY: CODETECH IT SOLUTION

NAME: MAYANK SAGAR

INTERN ID: CT04DN926

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

TASK DESCRIPTION: 
In Task 3 of the CodTech Data Science Internship, I was assigned the objective of building a machine learning model using real-world data and deploying it through a RESTful API using Flask. This task helped me combine core data science concepts with software engineering skills to create an end-to-end solution that can be used in real applications.

Objective:

The main goal of this task was to:

1. Load and preprocess a dataset.


2. Build and train a machine learning model.


3. Evaluate its performance.


4. Deploy the trained model using a Flask-based API.


5. Allow the model to accept input data as JSON and return predictions in real-time.




Step 1: Data Loading and Exploration

I began by loading the dataset (data.csv) using pandas. This dataset included various features related to employee attributes and the target variable Attrition, indicating whether an employee left the company or not. I used df.head() and df.info() to understand the structure and quality of the data. This step helped me identify which columns were numerical, which were categorical, and whether there were any missing values.


---

Step 2: Data Preprocessing

Before training the model, it was essential to preprocess the data:

Target and Features: I separated the target column Attrition from the features.

Encoding Categorical Variables: Since machine learning algorithms require numerical input, I applied One-Hot Encoding to convert categorical columns into binary vectors using pd.get_dummies().

Train-Test Split: I divided the dataset into training and testing subsets using an 80-20 split (train_test_split).

Feature Scaling: To normalize the feature range and improve model performance, I applied StandardScaler to scale the data.


These preprocessing steps were crucial to ensure the model could learn effectively without being biased by feature scales or data format inconsistencies.


---

Step 3: Model Training

I chose the Random Forest Classifier from sklearn.ensemble as the machine learning model because of its robustness, accuracy, and ability to handle both numerical and categorical features well.

Using the scaled training data, I trained the model using .fit() and then predicted the outcomes on the test data. The model showed promising results in terms of classification accuracy and other metrics.

 Step 4: Model Evaluation

I used classification_report from sklearn.metrics to evaluate the model. It gave me detailed insights into metrics like Precision, Recall, F1-Score, and Accuracy. These metrics helped me understand how well the model was able to predict employee attrition and where it might need improvement.


Step 5: Saving the Model and Preprocessing Tools

To ensure the trained model could be reused without retraining, I saved:

The trained Random Forest model (rf_model.pkl)

The Standard Scaler object (scaler.pkl)

The list of feature names (features.pkl)


These were saved using the joblib library, which is efficient for serializing large NumPy arrays and machine learning models.


Step 6: Creating the Flask API

To make the model accessible to other applications or users, I built a simple Flask API. The API had two main routes:

/: A welcome message

/predict: Accepts JSON input, processes it, scales it, and returns the predicted result


The prediction route:

Accepts a dictionary of feature values via POST request.

Converts it into a DataFrame.

Reorders columns to match training data and fills missing ones with 0.

Applies scaling using the saved scaler.

Uses the trained model to predict the result.


The API returns the prediction in JSON format, making it easy to integrate with web frontends or mobile apps.


Conclusion

Task 3 was extremely beneficial in bridging the gap between machine learning and deployment. It not only allowed me to apply my data analysis and modeling skills, but also introduced me to the practical side of deploying ML models using Flask. Through this task, I now understand the full pipeline — from raw data to a live API serving real-time predictions.

This experience was both challenging and rewarding, and I am confident that the skills I learned here will be highly valuable in real-world data science projects.
