# House-Price-Prediction with Machine Learning

# About the Project

This project aims to develop a machine learning model that accurately predicts housing prices using the Boston Housing dataset. By analyzing various features of houses, such as crime rate, number of rooms, and accessibility to highways, the model provides valuable insights for potential buyers or sellers in estimating housing prices. The project utilizes the powerful CatBoostRegressor algorithm for optimal performance and incorporates techniques like data preprocessing, exploratory data analysis, and model training. The trained model can be used as a tool to make informed decisions in the real estate market. 


## Built With

 - Flask
 - pandas
 - numpy
 - matplotlib
 - scikit-learn
 - catboost
 - gunicorn
 
 # Getting Started
This is make you understand how you may give instructions on setting up your project locally. To get a local copy up and running follow these simple example steps.

1. Clone the repo

```
git clone https://github.com/KalyanMurapaka45/Spam-Email-Detection.git
```

2. Install the required libraries

```
pip install -r requirements.txt
```

3. Open and execute .ipynb file (After complete Execution you will get a .pkl file for project Deployment)

# Dataset Description

## California Housing Dataset

The California Housing dataset is imported from the `sklearn.datasets` module in Python. It consists of a total of 506 instances, each representing a house in the California area. The dataset contains 13 numerical features that describe various aspects of the houses, such as crime rate, average number of rooms, and proximity to employment centers. The target variable is the median value of owner-occupied homes in thousands of dollars.

### Features

1. MedInc        median income in block group
2. HouseAge      median house age in block group
3. AveRooms      average number of rooms per household
4. AveBedrms     average number of bedrooms per household
5. Population    block group population
6. AveOccup      average number of household members
7. Latitude      block group latitude
8. Longitude     block group longitude

### Target Variable

- MEDV: Median value of owner-occupied homes in $1000s

# Data Preprocessing

The California Housing dataset is preprocessed before training the machine learning model. The dataset is imported and split into input features (X) and the target variable (y). The input features are then standardized using the `StandardScaler` from the `sklearn.preprocessing` module to ensure that all features have a similar scale. The preprocessed dataset is further divided into training and testing sets using a 80:20 train-test split ratio.

# Model Training and Evaluation

A CatBoostRegressor model is trained using the preprocessed dataset. The model is built to predict housing prices based on the given features. Hyperparameter tuning is performed using a RandomizedSearchCV approach from the `sklearn.model_selection` module. The best set of hyperparameters is selected based on 5-fold cross-validation. The model is trained on the training set using the optimized hyperparameters. The trained CatBoostRegressor model is evaluated using the testing set. The predicted housing prices are compared to the actual prices, and the performance of the model is assessed using the R-squared metric.

- Algorithm Used: ```Catboost Algorithm``` 

# Model Deployment

This project includes a Flask-based web application for deploying the house price prediction model. The model is loaded from the saved pickle file ('housepred.pkl'), and the scaler object is loaded from 'scaler.pkl' for preprocessing the input data. The web application allows users to input the necessary features of a house through a form or API request, and it returns the predicted house price.

# Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

# License

Distributed under the GNU General Public License v3.0. See ```LICENSE.txt``` for more information.

# Acknowledgements
This project was inspired by the Kaggle dataset on California House Price Prediction and the corresponding competition. We also acknowledge the open-source Python libraries used in this project and their contributors.