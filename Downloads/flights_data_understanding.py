#!/usr/bin/env python
# coding: utf-8

# In[40]:


#Q1
import pandas as pd

df = pd.read_csv("nycflights.csv")

newyork_flight_new = df[["dep_time", "dep_delay", "arr_time", "arr_delay", "tailnum"]]

print(newyork_flight_new.head())


# In[27]:


#Q2
dep_time_2000 = newyork_flight_new[newyork_flight_new["dep_time"] > 2000]

deleted_rows_count = len(newyork_flight_new) - len(dep_time_2000)

print(f"Number of deleted rows: {deleted_rows_count}")


# In[28]:


#Q3
import numpy as np

if newyork_flight_new["dep_delay"].isnull().any():
    median_dep_delay = newyork_flight_new["dep_delay"].median()
    
    newyork_flight_new["dep_delay"].fillna(median_dep_delay, inplace=True)
    
    print("Missing values in dep_delay were replaced with the median.")
else:
    print("No missing values found in dep_delay.")


# In[31]:


#Q4
filtered_flights = df.query("airtime > 120 and distance > 700")
print(filtered_flights)


# In[37]:


#Q5
df1 = newyork_flight_new.filter(["dep_time", "dep_delay", "arr_time", "arr_delay", "tailnum", "destination"])
print(df1)


# In[38]:


#Q6
df1 = df1.assign(total_delay=df1["dep_delay"] + df1["arr_delay"])
print(df1)


# In[39]:


#Q7
grouped_df = df.groupby("month").agg({"airtime": "mean", "distance": "max"})
print(grouped_df)


# In[11]:


#Q8
import pandas as pd

df = pd.read_csv("student-por.csv")

num_features = df.shape[1]

print("Number of features:", num_features)


# In[12]:


#Q9
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

regression_model_G1 = LinearRegression()
regression_model_G1.fit(df[["studytime"]], df['G1'])

regression_model_G2 = LinearRegression()
regression_model_G2.fit(df[["studytime"]], df['G2'])

predicted_G1 = regression_model_G1.predict(df[["studytime"]])
predicted_G2 = regression_model_G2.predict(df[["studytime"]])

mse_G1 = mean_squared_error(df['G1'], predicted_G1)
mse_G2 = mean_squared_error(df['G2'], predicted_G2)

print("Mean Squared Error (G1):", mse_G1)
print("Mean Squared Error (G2):", mse_G2)



# In the abive code, it is calculating the mean squared error, which will help in determining which is showing better performance. Lowest MSE shows the better performance. So, here G1 have better performance.

# In[13]:


#Q10
import pandas as pd
import matplotlib.pyplot as plt

plt.scatter(df['absences'], df['G1'])
plt.xlabel('Absences')
plt.ylabel('G1')
plt.title('Relationship between Absences and G1')
plt.show()

correlation_G1 = df['absences'].corr(df['G1'])
print("Correlation (Absences, G1):", correlation_G1)

plt.scatter(df['absences'], df['G2'])
plt.xlabel('Absences')
plt.ylabel('G2')
plt.title('Relationship between Absences and G2')
plt.show()

correlation_G2 = df['absences'].corr(df['G2'])
print("Correlation (Absences, G2):", correlation_G2)


# The correlation of G2 is slightly greater than G1, so G2 is showing strong linear re;ationship than G1
