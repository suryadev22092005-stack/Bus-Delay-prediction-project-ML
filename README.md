# Bus Delay Prediction Using Machine Learning
A machine learning project that predicts bus delay in minutes based on route, date, time, and other relevant features.

### 1. Demo

The application allows users to enter bus details and predict the expected delay.

![Bus Delay Prediction Demo](images/My%20project.gif)

### 2. Problem Statement and Impact

Bus delays can affect passengers, students, employees, and daily commuters.

The purpose of this project is to predict bus delay using machine learning techniques.

The prediction may help passengers understand the expected delay and support better travel planning.

### 3. Dataset

#### Dataset Source: Link: https://www.kaggle.com/datasets/karmansinghbains/ttc-delays-and-routes-2023[sd1.1]
#### (Note: Taken dataset is an Excel Format. Converted into CSV.)

#### Dataset Size: 
Columns->10 and Rows-> 50914

#### Features: 
Route,Day,Incident,Min Gap,Direction,Vehicle,Day_Date,Month,Year,Hour and	Minute.

#### Target Variable: 
Min Delay.

### 4. Approach

The project follows these steps:

Load the dataset.

Perform data cleaning.

Handle missing values.

Perform feature engineering.

Split the dataset into training and testing data.

Train machine learning models.

Evaluate model performance.

Select the best-performing model.

Deploy the model using Streamlit.

#### 5. Results

The following models were evaluated after hypertuned:

## Model Comparison

Before hypertuned:
| Model | MAE | MSE | R2 Score |
|---|---:|---:|---:|
| Linear Regression | 14.892246 | 3360.487695| 0.901925	|
| K-Nearest Neighbors| 45.082222 | 6255.441778 | 0.817436 |
| Random Forest | 12.851630 | 3326.890213 | 0.902905 |

After hypertuned:
| Model | MAE | MSE | R2 Score |
|---|---:|---:|---:|
| Linear Regression | 14.892246 | 3360.487695| 0.901925	|
| K-Nearest Neighbors| 43.973960 | 6130.739467 | 0.821075 |
| Random Forest | 12.135158 | 3093.946999 | 0.909704 |


### 6. Key Insights and Findings

Random Forest performed better than the other evaluated models.

Linear Regression provided a useful baseline.

Feature engineering helped prepare date and time information for prediction.

Model evaluation helped identify the most suitable model for deployment.

### 7. Challenges Faced

Handling missing values in the dataset.

Converting date and time into useful features.

Handling outliers.

Comparing different machine learning models.

Integrating the trained model into Streamlit.

Ensuring the same preprocessing steps are used during training and prediction.

### 8. How to Run Locally

* Clone the repository

* git clone YOUR_GITHUB_REPOSITORY_URL

* Navigate to the project folder

* cd bus-delay-prediction

* Install the required libraries

* pip install -r requirements.txt

* Run the Streamlit application

* streamlit run app.py

### 9. Limitations and Future Improvements

#### i) Limitations

* Predictions depend on the quality and availability of historical data.

* Unexpected traffic conditions may affect actual bus delays.

* Model predictions may not be accurate for routes or conditions not represented in the training data.

#### ii) Future Improvements

Collect more recent bus delay data.

Include traffic and weather information.

Improve feature engineering.

Evaluate additional machine learning models.

Improve the user interface.

### 10. Tech Stack

* Python

* Pandas

* NumPy

* Scikit-learn

* Matplotlib

* Streamlit

* Git and GitHub

### 11. Contact

Name: Surya Dev G.H.

LinkedIn:[Linked In](https://www.linkedin.com/in/surya-dev-gh/)

Email: [G-Mail](suryadev22092005@gmail.com)
