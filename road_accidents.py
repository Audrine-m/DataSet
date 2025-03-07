import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
data = pd.read_csv('road_accidents.csv')

# Display the first few rows
print(data.head())

# Define features (X) and target (y)
X = data[['Weather_Conditions', 'Road_Type', 'Speed_Limit', 'Vehicle_Type', 'Time_of_Day', 'Driver_Age', 'Alcohol_Involvement']]
y = data['Accident_Severity']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model to a file
joblib.dump(model, 'road_accident_severity_model.pkl')

# Load the saved model
loaded_model = joblib.load('road_accident_severity_model.pkl')

# Define a hypothetical set of independent variables
hypothetical_data = {
    'Weather_Conditions': 'rain',
    'Road_Type': 'highway',
    'Speed_Limit': 70,
    'Vehicle_Type': 'car',
    'Time_of_Day': 'night',
    'Driver_Age': 25,
    'Alcohol_Involvement': 1
}

# Convert the hypothetical data into a DataFrame
hypothetical_df = pd.DataFrame([hypothetical_data])

# One-hot encode the categorical variables
hypothetical_df = pd.get_dummies(hypothetical_df, drop_first=True)

# Ensure the columns match the training data
hypothetical_df = hypothetical_df.reindex(columns=X.columns, fill_value=0)

# Predict accident severity
predicted_severity = loaded_model.predict(hypothetical_df)
print(f"Predicted Accident Severity: {predicted_severity[0]}")


