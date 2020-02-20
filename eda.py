# Title: Delay prediction 
# Author: Sumaira Afzal 
# Script and data info: This script performs a time series analyses on ttc data.  
# Data consists of ttc delay.
# Data was collected by ttc between 2014 and 2019. 

import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv

df = read_csv("ttc_data.csv")

print(df.shape)

print(df.head(20))

print(df.describe())

#Feature Engineering
#Separate the Bounds (East, West,North,South)
#Separate Weekend Delays (Weekend Sat, Weend Sun)
#Isolating the month, year, and day of the week as separate features (Day,Month,Year)
#Add City Column to the dataset
#Get part of the day (Morning, Afternoon, Evening,Night)
#Add Hour and Minute column from Timestamp
#Add latitude and longitude (lat,lon)

#Separate bounds
df["East Bound"] = df["Bound"] == 'E'
df["West Bound"] = df["Bound"] == 'W'
df["North Bound"] = df["Bound"] == 'N'
df["South Bound"] = df["Bound"] == 'S'

# Separate Weekend Delays
df["Weekend-Sat"] = df["Day"] == 'Saturday'
df["Weekend-Sun"] = df["Day"] == 'Sunday'

#Isolating the month, year, and day of the week as separate predictors
#The numeric day of the year (ignoring the calendar year)
#if there were a seasonal component to these data, and it appears
#that there is, then the numeric day of the year would be best. Also, if some
#months showed higher success rates than others those should be taken into further EDA
#the month is preferable.

df.index = pd.to_datetime(df.index)
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Day']  = df.index.day

#add city column
df['City'] = 'TORONTO'

#add station location
df['Station_Location'] = df['Station'] + ' ' + df['City']

# split hour and minute
df[['Hour','Minute']] = df['Time'].astype(str).str.split(':', expand=True).astype(int)

#Create a function to get part of the day(morning, noon,afternoon evening and night) from Timestamp
def get_part_of_day(x):
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12 ):
        return 'Morning'
    elif (x > 12) and (x <= 16):
        return'Noon'
    elif (x > 16) and (x <= 20) :
        return 'Eve'
    elif (x > 20) and (x <= 24):
        return'Night'
    elif (x <= 4):
        return'Late Night'

df['Time-only'] = df['Hour'].apply(get_part_of_day)

# Convert the categorical variables to numerical
#Create a function to convert part of the day(morning, noon,afternoon evening and night) from Timestamp to numbers.

def convert_part_of_day(x1):
    if (x1 > 4) and (x1 <= 8):
        return 1
    elif (x1 > 8) and (x1 <= 12):
        return 2
    elif (x1 > 12) and (x1 <= 16):
        return 3
    elif (x1 > 16) and (x1 <= 20):
        return 4
    elif (x1 > 20) and (x1 <= 24):
        return 5
    elif ( x1 <= 4):
        return 6

df['Day-Half'] = df['Hour'].apply(convert_part_of_day)

#There are 475 stations so we have to encode the station location.
lb_station = LabelEncoder()
df["station_code"] = lb_station.fit_transform(df["Station_Location"])


#missing values
missing_values = df.isna().sum()
print(missing_values)
#Convert the Bounds into integer
df[['East Bound']] = df[['East Bound']].astype('int')
df[['West Bound']] = df[['West Bound']].astype('int')
df[['North Bound']] = df[['North Bound']].astype('int')
df[['South Bound']] = df[['South Bound']].astype('int')
df[['Weekend-Sat']] = df[['Weekend-Sat']].astype('int')
df[['Weekend-Sun']] = df[['Weekend-Sun']].astype('int')

#Create binary target variable (Delay) conditionally
df['Delay'] = [1 if x != 0 else 0 for x in df['Min Delay']]


#Feature Selection
df = df[['Year', 'Month','Day','Hour','Minute', 'Day-Half',  
         'East Bound', 'West Bound', 'North Bound', 'South Bound',
         'Weekend-Sat', 'Weekend-Sun', 'Station_Location', 'station_code', 'Min Gap', 'Delay']]

print(df.head(10))

df.to_csv("engineered_data.csv",index=False)