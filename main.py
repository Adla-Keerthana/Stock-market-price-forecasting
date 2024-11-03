# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# LOADING DATASET
data = pd.read_csv('C:/Users/T.Reddy/Downloads/archive (6)/data.csv')

# Information of data set
print(data.info())
# First 5 rows information
print(data.head())
# Columns information
print(data.columns)  
# Trim whitespace from column names
data.columns = data.columns.str.strip()

# DATA PREPROCESSING
# Check for missing values
print("Missing values in the dataset:")
print(data.isnull().sum())

# Preprocessing
# Assuming 'Close/Last', 'Volume', 'Open', 'High', 'Low' are your numeric columns
numeric_cols = ['Close/Last', 'Volume', 'Open', 'High', 'Low']

# Convert strings to floats
def convert_to_float(value):
    if isinstance(value, str):
        return float(value.replace('$', '').replace(',', ''))
    return float(value)

for col in numeric_cols:
    if col in data.columns:
        data[col] = data[col].apply(convert_to_float)

# Initialize the Standard Scaler
scaler = StandardScaler()

# Fit and transform the data
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Display only the first few rows of the preprocessed DataFrame
print("Preprocessed Data (first 5 rows):")
print(data.head())
print(data.dtypes)
print(data[numeric_cols].describe())

# VISUALIZING THE DATA
# Line Plot of Closing Prices
plt.figure(figsize=(12, 6)) 
for company in data['Company'].unique():  # Loop through each unique company in the dataset
    subset = data[data['Company'] == company]  # Create a subset for each company
    plt.plot(subset.index, subset['Close/Last'], label=company)  # Plot the closing prices
plt.title('Closing Prices Over Time')
plt.xlabel('Date')  # X-axis label
plt.ylabel('Close/Last Price')  # Y-axis label
plt.legend()  # Display the legend for different companies
plt.show()  # Show the plot

# Box Plot for Closing Prices
plt.figure(figsize=(12, 6)) 
sns.boxplot(x='Company', y='Close/Last', data=data)  # Create a box plot to show the distribution of closing prices by company
plt.title('Box Plot of Closing Prices by Company')
plt.ylabel('Close/Last Price')  # Y-axis label
plt.show()  

# Volume Distribution
plt.figure(figsize=(12, 6)) 
sns.histplot(data['Volume'], bins=30, kde=True)  # Creating a histogram of the volume with a kernel density estimate (KDE)
plt.title('Distribution of Volume')  
plt.xlabel('Volume')  # X-axis label
plt.ylabel('Frequency')  # Y-axis label
plt.show() 

# Correlation Analysis
plt.figure(figsize=(10, 6))  
# Only include numeric columns in the correlation calculation
correlation = data.select_dtypes(include=[np.number]).corr()  
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')  # Create a heatmap for the correlation matrix
plt.title('Correlation Heatmap')  
plt.show()  

# Daily Returns Calculation
data['Daily Return'] = data['Close/Last'].pct_change()  # Calculate daily returns as the percentage change in closing prices
plt.figure(figsize=(12, 6))
sns.histplot(data['Daily Return'].dropna(), bins=30, kde=True)  # Create a histogram of daily returns with KDE
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')  # X-axis label
plt.ylabel('Frequency')  # Y-axis label
plt.show()  

# FEATURE ENGINEERING
# Calculating Daily Returns
data['Returns'] = data['Close/Last'].pct_change()

# Calculate 5-day Simple Moving Average (SMA)
data['SMA'] = data['Close/Last'].rolling(window=5).mean()  # 5-day moving average

# Dropping NaN values generated from the moving average
data.dropna(inplace=True)

# Display the updated DataFrame
print("Data after Feature Engineering:")
print(data.head())

# DIVIDING DATA INTO X AND Y VARIABLES
X = data[['Returns', 'SMA']]
y = data['Close/Last']  # Change 'Close' to 'Close/Last'

# SPLITTING DATASET INTO TRAIN AND TEST DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STANDARDIZE FEATURES
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TRAINING THE MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# PREDICTING X_TEST DATA
y_pred = model.predict(X_test)

# MODEL EVALUATION MATRIX
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
predictions = pd.DataFrame({'Actual Prices': y_test, 'Predicted Prices': y_pred}, index=y_test.index)
print(predictions)

# PLOTTING PREDICTIONS VS ACTUAL
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Prices', color='red')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
