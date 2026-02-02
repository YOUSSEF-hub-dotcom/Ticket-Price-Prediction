
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import logging
logger = logging.getLogger(__name__)


def run_data_pipeline(file_path):
    df = pd.read_excel(file_path)
    logger.info("Loading Dataset ....")
    pd.set_option('display.width', None)

    print(df.head(30))

    logger.info("=========== Basic Functions ==========")
    logger.info("information about data:")
    print(df.info())

    logger.info("Statistical Operations:")
    print(df.describe())

    logger.info("Columns:")
    print(df.columns)

    logger.info("number of rows & columns:")
    print(df.shape)

    logger.info("Column types:")
    print(df.dtypes)

    logger.info("=========== Data Cleaning ==========")
    # ['Airline', 'Source', 'Destination', 'Arrival_Time', 'Duration',
    # 'Total_Stops', 'Additional_Info', 'Price', 'Date_of_Journey', 'Route', 'Dep_Time']

    logger.info("Validate DataType:")
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)

    # extract hours and minute from  Dep_Time
    df["Dep_hour"] = pd.to_datetime(df["Dep_Time"], format='%H:%M').dt.hour
    df["Dep_minute"] = pd.to_datetime(df["Dep_Time"], format='%H:%M').dt.minute

    # extract hours and minute from  Arrival_Time
    df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"].str.split(' ').str[0], format='%H:%M').dt.hour
    df["Arrival_minute"] = pd.to_datetime(df["Arrival_Time"].str.split(' ').str[0], format='%H:%M').dt.minute

    # Convert Duration just to minutes
    def convert_duration(duration):
        hours = 0
        minutes = 0

        parts = duration.split()

        for part in parts:
            if 'h' in part:
                hours = int(part.replace('h', ''))
            elif 'm' in part:
                minutes = int(part.replace('m', ''))

        return (hours * 60) + minutes

    df["Duration_mins"] = df["Duration"].apply(convert_duration)

    # Convert Total_Stops to Numeric
    stop_dict = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    df['Total_Stops'] = df['Total_Stops'].map(stop_dict)

    # After validate data type , now we don't Dep_Time & Arrival_Time because we extract minutes and hours from them
    df.drop('Dep_Time', axis=1, inplace=True)
    df.drop('Arrival_Time', axis=1, inplace=True)

    # and remove Duration because we extract it to minutes
    df.drop('Duration', axis=1, inplace=True)

    # Remove Route Column because we have Source & Destination Column
    df.drop('Route', axis=1, inplace=True)

    print(df['Additional_Info'].value_counts())
    # Remove Additional_Info Because most of its values No info > 80%
    # Therefore, it will not provide educational value to the model and may cause Overfitting , High variance, Noise.
    df.drop('Additional_Info', axis=1, inplace=True)

    logger.info("number of frequency rows")
    print(df.duplicated().sum())

    # we found 222 rows duplicated data this big problem (Overfitting ,Data Leakage ) so we must delete it
    df.drop_duplicates(inplace=True)
    print(f"Dataset shape after removing duplicates: {df.shape}")

    logger.info("missing values:")
    print(df.isnull().sum())

    mode_val = df["Total_Stops"].mode()[0]
    df["Total_Stops"] = df["Total_Stops"].fillna(mode_val)
    logger.info(f"Filled missing Total_Stops with mode: {mode_val}")

    print(df.isnull().sum())
    sns.heatmap(df.isnull(), cmap="YlOrRd")
    plt.title(" No Missing Values")
    plt.show()

    print(df.head(30))
    print(df.dtypes)

    logger.info("=========== Data Preprocessing ==========")

    # Price Skew
    skew_value = df['Price'].skew()
    logger.info("Skew Value of Price:", skew_value)

    sns.histplot(df['Price'], kde=True)
    plt.title("Distribution of Price Before Treatment Skew")
    plt.show()

    # Skew value of Price = 1.8 ---> we have Right-Skewed
    df['Price'] = np.log1p(df['Price'])

    treat_skew_price = df['Price'].skew()
    logger.info(f"Treatment Skew of Price:{ treat_skew_price}")
    # now it become : -2
    sns.histplot(df['Price'], kde=True)
    plt.title("Distribution of Price After Treatment Skew (Log Transformation)")
    plt.show()

    print("--------------------")

    # Price Outlier
    print(df['Price'].describe())
    # We noticed a difference between the max &75%
    # and the min &25% , which made us suspect there might be an outlier in the price column

    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)

    IQR = Q3 - Q1
    Lower = Q1 - 1.5 * IQR
    Upper = Q3 + 1.5 * IQR

    outliers = df[(df['Price'] < Lower) | (df['Price'] > Upper)]
    logger.info("Outliers Detect:", outliers)
    logger.info(f"Percentage of Outliers {len(outliers) / len(df) * 100} %")
    # Percentage of Outliers 0.08603383997705764 %

    sns.boxplot(df['Price'], color='blue')
    plt.title("BoxPlot to detect outlier in Price")
    plt.show()

    # we found 9 Rows in data contain outlier ----> So we Remove Them
    df = df[(df['Price'] >= Lower) & (df['Price'] <= Upper)]
    print(f"New Dataset shape: {df.shape}")

    print("--------------------")

    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
    df['Year_of_Journey'] = df['Date_of_Journey'].dt.year.nunique()
    df['Month_of_Journey'] = df['Date_of_Journey'].dt.month
    df['Days_of_Journey'] = df['Date_of_Journey'].dt.day
    df['Day_of_Week'] = df['Date_of_Journey'].dt.weekday
    df['Quarter'] = df['Date_of_Journey'].dt.quarter

    df.drop("Date_of_Journey", axis=1, inplace=True)

    if df['Year_of_Journey'].nunique() <= 1:
        df.drop("Year_of_Journey", axis=1, inplace=True)
        print("Year column removed because it has only one value.")

    df['is_weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 4 else 0)

    df['Path'] = df['Source'] + "-" + df['Destination']
    top_path = df['Path'].mode()[0]
    logger.info(f"Most frequent flight path: {top_path}")

    def assign_session(hour):
        if (hour >= 4) and (hour < 8):
            return 'Early Morning'
        elif (hour >= 8) and (hour < 12):
            return 'Morning'
        elif (hour >= 12) and (hour < 16):
            return 'Noon'
        elif (hour >= 16) and (hour < 20):
            return 'Evening'
        elif (hour >= 20) and (hour < 24):
            return 'Night'
        else:
            return 'Late Night'

    df['Dep_Session'] = df['Dep_hour'].apply(assign_session)

    df['Is_Long_Flight'] = df['Duration_mins'].apply(lambda x: 1 if x > 480 else 0)

    df['is_peak_season'] = df['Month_of_Journey'].apply(lambda x: 1 if x in [3, 5, 6, 12] else 0)

    print(df.head(30))

    return df

