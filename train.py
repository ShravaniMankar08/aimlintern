import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data
data = {
    "area": [500, 700, 900, 1100, 1300],
    "price": [20, 28, 36, 45, 52]
}

df = pd.DataFrame(data)

# Input and output
X = df[["area"]]
y = df["price"]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict price for 1000 sq ft
prediction = model.predict([[1000]])

print("Predicted price:", prediction[0])