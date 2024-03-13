from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import fabs
import numpy as np

def evaluate_model(y_test, y_pred):
  """
  This function calculates various metrics to evaluate the LSTM model's performance.

  Args:
      y_test (list): List of actual target values.
      y_pred (list): List of predicted target values.

  Returns:
      dict: Dictionary containing the calculated metrics (MAE, MSE, R-squared, SMAPE, KGE).
  """
  # Mean Absolute Error (MAE)
  mae = mean_absolute_error(y_test, y_pred)

  # Mean Squared Error (MSE)
  mse = mean_squared_error(y_test, y_pred)

  # R-squared

  r2 = r2_score(y_test, y_pred)

  # Symmetric Mean Absolute Percentage Error (SMAPE)
  smape = 0
  for i in range(len(y_test)):
    if y_test[i] == 0:
      continue  # Avoid division by zero
    smape += fabs((y_pred[i] - y_test[i]) / (abs(y_test[i]) + abs(y_pred[i]))) / len(y_test)
  smape *= 2 * 100  # Convert to percentage

  # Kling-Gupta Efficiency (KGE)
  def mean(arr):
    return sum(arr) / len(arr)

  y_bar = mean(y_test)
  sigma_y = np.std(y_test)
  sigma_p = np.std(y_pred)
  r = np.corrcoef(y_test, y_pred)[0, 1]
  alpha = 1  # Weighting factor for mean relative error (can be adjusted)
  kge = 1 - np.sqrt( 
      (mean((y_test - y_pred) ** 2) / (sigma_y ** 2)) + 
      (fabs(mean(y_test) - mean(y_pred)) / y_bar) ** alpha + 
      (fabs(sigma_y - sigma_p) / sigma_y) ** 2
  )

  return {
      "MAE": mae,
      "MSE": mse,
      "R-squared": r2,
      "SMAPE": smape,
      "KGE": kge
  }

# # Call the function after your model prediction
# metrics = evaluate_model(y_test, y_pred)
# print(metrics)