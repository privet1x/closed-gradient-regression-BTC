# pylint: disable=invalid-name,redefined-outer-name,redefined-builtin,import-error
# mypy: ignore-errors

"""
This module performs regression analysis on a given dataset. It can handle both
linear and polynomial regression using closed-form solutions and gradient descent.
"""
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error  # Unused import as per pylint

# Define constants for magic numbers
FIGURE_SIZE = (10, 6)
NUM_POINTS = 500

# Define a function to read and preprocess the data
def read_and_preprocess(file_path):
    """
    Reads the dataset from a CSV file, handles missing values and extracts relevant features.
    Args:
    - file_path (str): The path to the CSV file containing the dataset.

    Returns:
    - pandas.DataFrame: A DataFrame with Timestamps converted to a more useful format
    and missing values handled.
    """
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Timestamp'], unit='s')
    # Using 'Weighted_Price' as target and Date as feature
    data = data[['Date', 'Weighted_Price']].dropna()  # Removing rows with NaN in 'Weighted_Price'
    data.set_index('Date', inplace=True)
    # Transform date to numerical feature: days since start
    data['Time'] = (data.index - data.index[0]).days
    return data

# Define a function for the closed form solution
def closed_form_solution(features, targets):
    """
    Solves linear regression using the closed form solution.
    Args:
    - features (np.array): The feature matrix (including bias).
    - targets (np.array): The target variable.

    Returns:
    - np.array: The coefficients of the regression.
    """
    # Adding bias term
    features_with_bias = np.c_[np.ones((features.shape[0], 1)), features]
    # Normal equation
    coefficients = np.linalg.inv(features_with_bias.T.dot(features_with_bias)).dot(features_with_bias.T).dot(targets)
    return coefficients

# Define a function for polynomial closed form solution
def polynomial_closed_form(features, targets, degree):
    """
    Extends features to a polynomial and then applies the closed form solution for linear regression.
    Args:
    - features (np.array): The original feature matrix.
    - targets (np.array): The target variable.
    - degree (int): The degree of the polynomial features.

    Returns:
    - np.array: The coefficients of the polynomial regression.
    """
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    features_poly = poly_features.fit_transform(features.reshape(-1, 1))
    features_poly_with_bias = np.c_[np.ones((features_poly.shape[0], 1)), features_poly]
    coefficients_poly = np.linalg.inv(features_poly_with_bias.T.dot(features_poly_with_bias)).dot(features_poly_with_bias.T).dot(targets)
    return coefficients_poly

# Define a function to plot the regression
def plot_regression(X, y, model, degree, file_name):
    """
    Plots the datapoints and the regression line or curve.
    Args:
    X (np.array): The feature data.
    y (np.array): The target data.
    model (np.array or object): The model coefficients or a sklearn model.
    degree (int): Degree of the polynomial used; 1 for linear.
    file_name (str): The file path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data points')
    X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)

    if degree > 1:
        poly_features = PolynomialFeatures(degree=degree)
        X_plot = poly_features.fit_transform(X_plot)

    # Check if the model is a sklearn model with predict method
    if hasattr(model, 'predict'):
        y_plot = model.predict(X_plot)
    else:  # Otherwise, it's assumed to be an array of coefficients
        if degree == 1:  # If linear regression, add bias term
            X_plot = np.insert(X_plot, 0, 1, axis=1)
        y_plot = X_plot.dot(model)

    plt.plot(X_plot[:, 1], y_plot, color='red', label='Regression fit')  # X_plot[:, 1] if degree > 1 else X_plot
    plt.xlabel('Days since start')
    plt.ylabel('Weighted Price')
    plt.legend()
    plt.title('Regression Analysis')
    plt.savefig(file_name)
    plt.close()


# The main block to process command line arguments and run the regression analysis
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python miw_s22326_task03.py <input_file> <algorithm_type> <polynomial_degree>")
        sys.exit(1)

    input_file, algorithm_type, degree = sys.argv[1:4]
    degree = int(degree)

    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = f'results_{algorithm_type}_degree_{degree}_{timestamp}.txt'
    plot_file = f'regression_plot_{algorithm_type}_degree_{degree}_{timestamp}.png'

    data = read_and_preprocess(input_file)
    X, y = data['Time'].values, data['Weighted_Price'].values

    if algorithm_type.lower() == 'closed':
        if degree == 1:
            model = closed_form_solution(X, y)
        else:
            model = polynomial_closed_form(X, y, degree)
        plot_regression(X, y, model, degree, plot_file)
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(f'Coefficients: {model}\n')
    elif algorithm_type.lower() == 'gradient':
        model = LinearRegression(fit_intercept=False)
        if degree == 1:
            model.fit(X.reshape(-1, 1), y)
        else:
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X.reshape(-1, 1))
            model.fit(X_poly, y)
        plot_regression(X, y, model, degree, plot_file)
        with open(output_file, 'w', encoding='utf-8') as file:
            coef = np.concatenate(([model.intercept_], model.coef_))
            file.write(f'Coefficients: {coef}\n')
    else:
        print("Invalid algorithm type specified. Choose 'closed' or 'gradient'.")
        sys.exit(1)
