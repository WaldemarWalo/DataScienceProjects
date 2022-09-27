# Title: Price Prediction for used Cars

## Brief Summary:
- Price prediction for selected model and make from a dataset of used cars for 8 makes and 196 car models. The full dataset contains 100k car listings and 9 features.
- Trained Linear Regression, Lasso and Ridge with Polynomial Features selected with SelectKBest produced an R2 score of 0.873. Achieved the best R2 score of 0.885 with CatBoostRegressor


## Description
The goal of this project is to predict the price of a car based on 9 features. The dataset has a very limited number of features and is missing essential information about car trim, ownership type, and accident history.

## Dataset
- this is a public Dataset available on [Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)s
- the combined dataset contains 100k listings of cars from the UK and includes 8 makes and 196 models with diesel, petrol and hybrid fuel types.
- available features: 'make', 'model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', and 'engineSize' with regression target 'price'

## EDA
- each make contains between 4711 (Hyundai) and 17751 (Ford) listings.
- across all makes, 132 models have less than 10 listings.
- the dataset contains 1475 duplicate
- the 'tax' column for Hyundai had a different name
- there are no missing values
- mpg has many invalid values, especially around hybrids and electric vehicles

## Transformations:
- Unify the 'tax' column within makes
- Drop 1475 duplicates
- Strip whitespaces from model names
- drop 9 listings with 'transmission' = 'Other'
- drop 252 listings with 'fuelType' = 'Other' or 'Electric'
- drop 265 listings with 'engineSize'] = 0
- drop cars produced before 1990 and with invalid production year > 2022

## Feature Engineering:
- the limited number of features in this dataset does not provide the opportunity for extensive feature engineering
- I've created the 'car' feature that represents a combined make and model
- the Year column was changed to age
- I've used Ordinal Encoding for fuel type and transmission
- Training one model for all 196 unique car types will require creating 196 one-hot encoded features. For this dataset, I've decided to select a few cars (make-model combination) and create models for them individually

# Cross Validation
- I've used RepeatedKFold with 6 splits and 3 repeats to shortlist the models and narrow down hyperparameters, and later increased the splits to 8 and repeats to 5 for the final run

# Models:
- I've utilized the PolynomialFeatures to create a new feature matrix with 2 to 5 degrees
- new features were scaled and passed to SelectKBest to select up to 200 best features
- evaluated Linear Regression, Lasso and Ridge with multiple alpha values. Ridge gave the best R2 score of 0.864
- evaluated multiple tree models and achieved an R2 score of 0.885 with CatBoostRegressor