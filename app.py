import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# ==========================
# 1. Load Dataset
# ==========================
df = pd.read_csv("dataset/retail_sales.csv")

print("First 5 Rows")
print(df.head())

# ==========================
# 2. Data Preprocessing
# ==========================

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Sort by date
df.sort_values('date', inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# ==========================
# 3. Feature Engineering
# ==========================

df['Day'] = df['date'].dt.day
df['Month'] = df['date'].dt.month
df['Year'] = df['date'].dt.year

# ==========================
# 4. Prepare Features
# ==========================

X = df[['Stock', 'Price', 'Day', 'Month', 'Year']]
y = df['Sales']

# ==========================
# 5. Train-Test Split
# ==========================

split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ==========================
# 6. Train Model
# ==========================

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================
# 7. Prediction
# ==========================

y_pred = model.predict(X_test)

# ==========================
# 8. Evaluation
# ==========================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print("MAE :", round(mae,2))
print("MSE :", round(mse,2))
print("RMSE:", round(rmse,2))
print("R2 Score:", round(r2,4))

# ==========================
# 9. Predict Future Sales
# ==========================

last = df.iloc[-1]

future_dates = pd.date_range(
    start=last['date'] + pd.Timedelta(days=1),
    periods=30
)

future = pd.DataFrame({
    "Stock":[last['Stock']]*30,
    "Price":[last['Price']]*30
})

future['Day'] = future_dates.day
future['Month'] = future_dates.month
future['Year'] = future_dates.year

future_sales = model.predict(future)

future_df = pd.DataFrame({
    "Date":future_dates,
    "Predicted Sales":future_sales
})

print("\nFuture Sales Prediction")
print(future_df)

# ==========================
# 10. Visualization
# ==========================

plt.figure(figsize=(12,5))
plt.plot(df['date'],df['Sales'],label="Actual Sales")
plt.title("Historical Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(y_test.values,label="Actual")
plt.plot(y_pred,label="Predicted")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Samples")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(future_df['Date'],future_df['Predicted Sales'],color="green")
plt.title("Next 30 Days Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.grid()
plt.show()
import joblib

joblib.dump(model, "sales_model.pkl")
model = joblib.load("sales_model.pkl")