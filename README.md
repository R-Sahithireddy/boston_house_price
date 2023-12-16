# END TO END BOSTON HOUSE PRICE PREDICTION 

## Problem Statement: 
The problem revolves around predicting house prices in Boston using various features. The dataset contains information about different factors that might influence house prices, such as the average number of rooms per dwelling, pupil-teacher ratio, and the percentage of the population considered lower status.

## Description:
The Boston House Price Prediction problem involves developing a model to estimate housing prices based on multiple features.

## Exploratory Data Analysis (EDA) Conclusion:

Upon performing EDA, several key observations and conclusions that can be drawn are:

### Outliers:
From the analysis, it's apparent that outliers exist in the data. These outliers might adversely affect the performance of a linear regression model.

### Feature Relationships:
Most features showcase a linear relationship with the target variable 'MEDV'. Areas with a higher percentage of lower-status population tend to have lower house prices. Additionally, houses with more rooms demonstrate a positive correlation with higher prices. The pupil-teacher ratio alone may not be a strong predictor of house prices in the dataset.

## Model Implementation:

The predictive model is implemented in the `src` folder, specifically in the `boston.py` file. The model leverages functions for enhanced modularity.

## Project Structure:

`src/boston.py`: Contains the implemented predictive model using functions.<br>
`src/exceptions.py`: Exception handling function for reuse in the code.<br>
`src/logger.py`: Logging configuration for improved debugging.<br>
`src/utils.py`: General utilities for reuse in the code.<br>

## References:

Dataset Source, "Boston Housing Dataset," https://www.kaggle.com/datasets/vikrishnan/boston-house-prices.


