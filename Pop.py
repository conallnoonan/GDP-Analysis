import pandas as pd # NOQA F401
import numpy as np # NOQA F401
import seaborn as sns  # NOQA F401
import matplotlib.pyplot as plt # NOQA F401
import plotly.express as px # NOQA F401
from sklearn.model_selection import train_test_split as tts # NOQA F401
from sklearn.preprocessing import StandardScaler # NOQA F401
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor # NOQA F401
from sklearn.datasets import make_regression # NOQA F401
from sklearn.tree import DecisionTreeRegressor  # NOQA F401
from sklearn.ensemble import RandomForestRegressor  # NOQA F401
from sklearn.svm import SVR  # NOQA F401
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # NOQA F401

df = pd.read_csv("/Users/conallnoonan/Documents/GDP_Analysis/Pop_2024.csv")
print(df.head())
print(df.info())


def convarea(area):     # Converting area to float from object
    area = area.strip().replace("<", "").strip()
    # removing '<' characters
    if 'M' in area:
        area = float(area.replace("M", "")) * 1000000
    elif 'K' in area:
        area = float(area.replace("K", "")) * 1000
    else:
        area = float(area)
    return area


df["Area (km2)"] = df["Area (km2)"].apply(convarea)
print(df.head())

df.fillna(0, inplace=True)  # Filling NaN values with 0
print(df.isnull().sum())  # Checking for null values


# Visualizations
# Growth Rates of Countries according to Population(Top 10)
fig = px.bar(df[0:10], x="Country", y="Population 2024",
             color="Growth Rate", barmode="relative")
fig.show()
fig.write_html("growth_chart(10).html", auto_open=True)

# Growth Rates of Countries according to Population(Top 100
fig = px.bar(df[0:100], x="Country", y="Population 2024",
             color="Growth Rate", barmode="relative")
fig.show()
fig.write_html("growth_chart(100).html", auto_open=True)

# Population Change

df["population change"] = df["Population 2024"] - df["Population 2023"]
fig = px.bar(df.nlargest(10, "population change").
             sort_values(by='population change'), x="population change",
             y="Country", color="Growth Rate", barmode="relative")
fig.show()
fig.write_html("Pop_change(10).html", auto_open=True)

# Scatter Plot of Growth Rate vs World Rank
fig = px.scatter(df, x="Growth Rate", y="World Rank",
                 title="Growth Rate vs World Rank",
                 color="Density (/km2)",
                 hover_name="Country",
                 hover_data=['Population 2024', 'Population 2023', ''
                             'Area (km2)', 'Density (/km2)', 'World %'],
                 template="plotly_dark")
fig.show()
fig.write_html("Growth Rate vs World Rank.html", auto_open=True)


# Distribution of Population
plt.figure(figsize=(10, 6))
sns.histplot(df["Population 2024"], bins=35, kde=True)
plt.title("Distribution of Population in 2024")
plt.xlabel("Population 2024")
plt.ylabel("Frequency")
plt.savefig("/Users/conallnoonan/Documents/GDP_Analysis/pop_distribution.png")
plt.close()

# Correlation Heatmap
numericvalues = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numericvalues.corr(), annot=True, fmt=".2f")
plt.title("Population Correlation Heatmap")
plt.savefig("/Users/conallnoonan/Documents/GDP_Analysis/pop_heatmap.png")
plt.close()

# Predictions and Regression Models

df.info()

# Train-test split
x = df[["Population 2023", "Area (km2)", "Density (/km2)", "Growth Rate",
        "population change"]]
y = df["Population 2024"]
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2,
                                       random_state=42)
scaler = StandardScaler()
x_trainscaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Linear Regression
model1 = LinearRegression()
model1.fit(x_trainscaled, y_train)
y_pred1 = model1.predict(x_test_scaled)
print("------------------------------------")
print("Linear Regression:")
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred1))
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred1))
print("R^2 Score: ", r2_score(y_test, y_pred1))
print("------------------------------------")

# Decision Tree
model2 = DecisionTreeRegressor()
model2.fit(x_trainscaled, y_train)
ypred2 = model2.predict(x_test_scaled)
print("------------------------------------")
print("Decision Tree : ")
print("Mean Squared Error: ", mean_squared_error(y_test, ypred2))
print("Mean Absolute Error: ", mean_absolute_error(y_test, ypred2))
print("R^2 Score: ", r2_score(y_test, ypred2))
print("------------------------------------")
# Random Forest
model3 = RandomForestRegressor()
model3.fit(x_trainscaled, y_train)
ypred3 = model3.predict(x_test_scaled)
print("------------------------------------")
print("Random Forest : ")
print("Mean Squared Error: ", mean_squared_error(y_test, ypred3))
print("Mean Absolute Error: ", mean_absolute_error(y_test, ypred3))
print("R^2 Score: ", r2_score(y_test, ypred3))
print("------------------------------------")

# Saving as new CSV
y_pred_all = model1.predict(scaler.transform(x))
y_pred_all_series = pd.Series(y_pred_all, index=x.index,
                              name="Predicted_Pop_2024_LR")
dfnew = pd.concat([df["Country"], x, y, y_pred_all_series], axis=1)
dfnew.to_csv(
    "/Users/conallnoonan/Documents/GDP_Analysis/Pop_Predictions_L.csv",
    header=True, index=False)
