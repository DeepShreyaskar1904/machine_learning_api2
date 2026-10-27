from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
hours = np.array([[1],[2],[3],[4],[5],[6]])
score = np.array([5,15,29,32,38,44])
model = LinearRegression()
model.fit(hours,score)
joblib.dump(model,'test_ml_api.joblib')
print("Model saved successfully!")