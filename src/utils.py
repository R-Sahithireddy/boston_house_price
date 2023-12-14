import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def scatter_plot(y_test, y_pred):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred)
    plt.plot([0, 50], [0, 50], "--k")
    plt.xlabel("Actual House Prices ($1000)")
    plt.ylabel("Predicted House Prices: ($1000)")
    plt.title("Actual Prices vs Predicted prices")
    plt.savefig("scatter_plot.png")
    plt.show()


def box_plot(pred):
    figure = plt.figure()
    figure.suptitle("Linear Regression Box Plot")
    plt.boxplot(pred)
    plt.xlabel(pred, rotation=45, ha="right")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.margins(0.05, 0.1)
    plt.savefig("box_plot.png")
    plt.show()
