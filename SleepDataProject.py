import numpy as np 
import matplotlib.pyplot as mp 
import pandas as pd
from datetime import datetime 
import math
import seaborn as sns
from sklearn.linear_model import LinearRegression
import random

df = pd.read_csv('/Users/thomassilva/Downloads/Autosleep.csv')
print(df.info())

#show the statistical summary on the numeric columns 
print(df.describe())

#find the least and most time spent asleep in one session
print("The most time alseep is:",df['asleep'].max())
print("The least amount of time alseep is:",df['asleep'].min())

# Convert the time values to timedelta data type
df['asleep'] = pd.to_timedelta(df['asleep'])
df['deep'] = pd.to_timedelta(df['deep'])

# Convert the timedelta values to float values representing the number of seconds
df['sleep_seconds'] = df['asleep'].dt.total_seconds().astype(float)
df['deep_sleep_seconds'] = df['deep'].dt.total_seconds().astype(float)

# Convert the float values to hours by dividing by 3600
df['sleep_hours'] = df['sleep_seconds'] / 3600
df['deep_sleep_hours'] = df['deep_sleep_seconds'] / 3600

# Find the mean of the 'sleep_hours' column
mean_sleep_hours = df['sleep_hours'].mean()
mean_deep_sleep_hours = df['deep_sleep_hours'].mean()
print("The average time alseep is:", mean_sleep_hours, "hours")
print("The average time in deep sleep is:", mean_deep_sleep_hours, "hours")

#dropping unnecessary columns (all of these have no values recorded)
df.drop(labels=['SpO2Avg', 'SpO2Min', 'SpO2Max', 'respAvg', 'respMin', 'respMax', 'tags', 'notes'], axis=1, inplace=True)

# altering the ISO8601 column to display the short date instead of the ISO date
df['ISO8601'] = df['ISO8601'].apply(lambda x: x.split('T')[0])
df = df.rename(columns={'ISO8601': 'short_date'})
print(df)

# generating a pivot table that shows the maximum time asleep for a given sleep efficiency score
pivot_data = df[df.sessions == 1].pivot_table(index='efficiency', values='asleep', aggfunc=max) 
print(pivot_data)

# generating a pivot table that shows the maximum deep sleep time for a given sleep efficiency score
df['deep'] = pd.to_timedelta(df['deep'])
pivot_data_deep = df.pivot_table(index='efficiency', values='deep', aggfunc=max) 
print(pivot_data_deep)

# Calculate sleep efficiency
df['sleep_efficiency'] = df['asleep'] / df['inBed']
print(df.head())

# Calculate the percentile rank of each night's sleep duration
df['asleep'] = pd.to_numeric(df['asleep'])
df['time_asleep_percentile'] = df['asleep'].rank(pct=True)
print(df.head())

# Print the correlation between deep sleep and sleep efficiency
df['deep'] = pd.to_numeric(df['deep']) / 3600
corr = df['deep'].corr(df['efficiency'])
print(df['deep'])
# print(corr)

# print the correlation between time asleep and all other columns
df['asleep'] = pd.to_numeric(df['asleep']) / 3600
df['asleep'] = df['asleep'].round(2)
correlations = df.corrwith(df['asleep'])
print(correlations)

# print the correlation between deep sleep and all other columns 
df['deep'] = pd.to_numeric(df['deep'])/ 3600
correlations1 = df.corrwith(df['deep'])
print(correlations1)

# Find the minimum and maximum values of the correlation and include the column name for the min and max values 
min_value = math.inf
max_value = -math.inf
for col, value in correlations.items():
    if value < min_value:
        min_value = value
        min_column = col
    if value > max_value:
        max_value = value
        max_column = col

print("Minimum correlation:", min_value, "in column", min_column)
print("Maximum correlation:", max_value, "in column", max_column)

# Plot and regression line
df['sleep_hours'] = round(df['sleep_hours'], 2)
sns.scatterplot(x=df['sleep_hours'].round(2), y=df['hrv'].round(2), data=df)
sns.regplot(x=df['sleep_hours'].round(2), y=df['hrv'].round(2), data=df, color='red')
mp.title("Sleep hours vs HRV")
mp.xlabel("Sleep hours")
mp.ylabel("HRV")
# mp.show()
print(df['sleep_hours'].describe())

# Grouping the data by month 
df['short_date'] = pd.to_datetime(df['short_date'])
df['month'] = df['short_date'].dt.month
mean_sleep_by_month = df.groupby('month')['sleep_hours'].mean()
mean_deep_sleep_by_month = df.groupby('month')['deep_sleep_hours'].mean()

# Creating a figure 
fig, ax = mp.subplots()
ax.set_title('Average Sleep and Deep Sleep by Month')
mean_sleep_by_month.plot(kind='bar', color='blue', ax=ax, label='Sleep')
mean_deep_sleep_by_month.plot(kind='bar', color='green', ax=ax, label='Deep Sleep')
ax.legend()
mp.show()

# Using the monthly data to create a model 
X = pd.DataFrame({'month': mean_sleep_by_month.index, 'mean_sleep_hours': mean_sleep_by_month.values})
y = pd.Series(mean_deep_sleep_by_month.values, name='mean_deep_sleep_hours')
model = LinearRegression()
model.fit(X, y)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
