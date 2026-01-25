import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    print("=========== EDA & Visualization ==========")

    print("What is the distribution of airfare prices?")
    print(df['Price'].describe())

    plt.figure(figsize=(8,5))
    sns.histplot(df['Price'], bins=50, kde=True)
    plt.title("Distribution of Flight Ticket Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.show()

    print("--------------------------")
    print("Does the ticket price vary depending on the airline?")
    ticket_price_per_Airline = df.groupby('Airline')['Price'].agg(['mean', 'median', 'std']).sort_values(by='mean', ascending=False)
    print(ticket_price_per_Airline)

    plt.figure(figsize=(10,5))
    sns.boxplot(x='Airline', y='Price', data=df, color='blue')
    plt.title("Ticket Price Distribution by Airline")
    plt.xticks(rotation=45)
    plt.show()

    print("--------------------------")
    print("Are there airlines whose prices are noticeably higher than others?")
    Avg_Price_Airline = df.groupby('Airline')['Price'].mean().sort_values(ascending=False)
    print(Avg_Price_Airline)

    plt.figure(figsize=(10,5))
    Avg_Price_Airline.plot(kind='barh')
    plt.title("Average Ticket Price per Airline")
    plt.xlabel("Average Price")
    plt.ylabel("Airline")
    plt.show()

    print("--------------------------")
    print("Is the price difference within the same airline significant?")
    STD_Price_Airline = df.groupby('Airline')['Price'].std().sort_values(ascending=False)
    print(STD_Price_Airline)

    plt.figure(figsize=(10,5))
    sns.violinplot(x='Airline', y='Price', data=df)
    plt.title("Price Variability Within Each Airline")
    plt.xticks(rotation=45)
    plt.show()

    print("--------------------------")

    print("Does the departure city(Source) affect the ticket price?")
    Avg_Price_Source = df.groupby('Source')['Price'].mean().sort_values(ascending=False)
    print(Avg_Price_Source)

    plt.figure(figsize=(8,5))
    Avg_Price_Source.plot(kind='bar', color='blue')
    plt.title("Ticket Price by Source City")
    plt.show()

    print("--------------------------")
    print("Does the arrival city(Destination) affect the ticket price?")
    Avg_Price_Destination = df.groupby('Destination')['Price'].mean().sort_values(ascending=False)
    print(Avg_Price_Destination)

    plt.figure(figsize=(8,5))
    sns.boxplot(x='Destination', y='Price', data=df, color='green')
    plt.title("Ticket Price by Destination City")
    plt.show()

    print("--------------------------")
    print("Does the ticket price vary depending on the number of stops?")
    TotalStops_Price_of_Ticket = df.groupby('Total_Stops')['Price'].agg(['mean', 'median', 'std'])
    print(TotalStops_Price_of_Ticket)

    plt.figure(figsize=(8,5))
    sns.barplot(x='Total_Stops', y='Price', data=df, color='Maroon')
    plt.title("Ticket Price by Number of Stops")
    plt.xlabel("Total Stops")
    plt.ylabel("Price")
    plt.show()

    print("--------------------------")

    print("Does the departure time(Dep_hour) affect the ticket price?")
    Dep_hour_affectOnPrice = df.groupby('Dep_hour')['Price'].mean().head()
    print(Dep_hour_affectOnPrice)

    plt.figure(figsize=(10,5))
    sns.lineplot(x='Dep_hour', y='Price', data=df, estimator='mean', color='coral')
    plt.title("Average Price vs Departure Hour")
    plt.grid()
    plt.show()

    print("--------------------------")
    print("Are morning flights more expensive than night flights?")
    Deep_Session_per_price = df.groupby('Dep_Session')['Price'].mean().sort_values(ascending=False)
    print(Deep_Session_per_price)

    plt.figure(figsize=(8,5))
    sns.barplot(x='Dep_Session', y='Price', color='teal', data=df)
    plt.title("Average Price by Departure Session")
    plt.show()

    print("--------------------------")

    print("Does the arrival time affect the price?")
    Arrival_time_affectOnPrice = df.groupby('Arrival_hour')['Price'].mean().head()
    print(Arrival_time_affectOnPrice)

    plt.figure(figsize=(10,5))
    sns.lineplot(x='Arrival_hour', y='Price', data=df, estimator='mean', color='purple')
    plt.title("Average Price vs Arrival Hour")
    plt.grid()
    plt.show()

    print("--------------------------")
    print('Does the duration of the trip affect the ticket price?')
    print(df[['Duration_mins','Price']].corr())

    plt.figure(figsize=(8,5))
    sns.scatterplot(x='Duration_mins', y='Price', data=df, alpha=0.5)
    plt.title("Price vs Flight Duration")
    plt.show()

    print("--------------------------")
    print('Is the relationship between flight duration and price linear?')
    print(df[['Duration_mins','Price']].corr(method='pearson'))

    plt.figure(figsize=(8,5))
    sns.regplot(x='Duration_mins', y='Price', data=df, scatter_kws={'alpha':0.3})
    plt.title("Linear Relationship Between Duration and Price")
    plt.show()

    print("--------------------------")
    print('Are long flights always more expensive than short flights?')
    print(df.groupby('Is_Long_Flight')['Price'].mean())

    plt.figure(figsize=(6,5))
    sns.boxplot(x='Is_Long_Flight', y='Price', data=df, color='purple')
    plt.title("Price Comparison: Long vs Short Flights")
    plt.show()

    print("--------------------------")

    print('Does the ticket price vary depending on the day of the trip?')
    print(df.groupby('Day_of_Week')['Price'].mean())

    plt.figure(figsize=(8,5))
    sns.barplot(x='Day_of_Week', y='Price', data=df)
    plt.title("Average Price by Day of Week")
    plt.show()

    print("--------------------------")

    print('Are weekend trips more expensive than other days?')
    print(df.groupby('is_weekend')['Price'].mean())

    plt.figure(figsize=(6,5))
    sns.boxplot(x='is_weekend', y='Price', data=df)
    plt.title("Weekend vs Weekday Prices")
    plt.show()

    print("--------------------------")

    print('Does the ticket price vary from month to month?')
    print(df.groupby('Month_of_Journey')['Price'].mean())

    plt.figure(figsize=(10,5))
    sns.lineplot(x='Month_of_Journey', y='Price', data=df, estimator='mean')
    plt.grid()
    plt.title("Average Price by Month")
    plt.show()

    print("--------------------------")

    print('Does the peak season lead to higher prices?')
    print(df.groupby('is_peak_season')['Price'].mean())

    plt.figure(figsize=(6,5))
    sns.boxplot(x='is_peak_season', y='Price', data=df)
    plt.title("Peak Season vs Non-Peak Season Prices")
    plt.show()


    print("--------------------------")

    print("Does the airline's impact on price vary depending on the number of stops?")
    Airline_Price_Total_Stops = pd.pivot_table(df, values='Price', index='Airline', columns='Total_Stops', aggfunc='mean')
    print(Airline_Price_Total_Stops)

    plt.figure(figsize=(10,6))
    sns.heatmap(
        Airline_Price_Total_Stops, annot=True, fmt=".0f", cmap="coolwarm"
    )
    plt.title("Airline vs Stops (Average Price)")
    plt.show()

    print("--------------------------")

    print('Does the effect of the departure time differ between days of the week?')
    Dep_hour_Day_of_Week_onPrice = pd.pivot_table(df, values='Price', index='Dep_hour', columns='Day_of_Week', aggfunc='mean').head()
    print(Dep_hour_Day_of_Week_onPrice)

    plt.figure(figsize=(10,6))
    sns.heatmap(
        Dep_hour_Day_of_Week_onPrice,
        cmap='viridis'
    )
    plt.title("Departure Hour vs Day of Week (Price)")
    plt.show()

    print("--------------------------")

    print('Does the price of the route change during peak season compared to regular seasons?')
    Path_is_Peak_Session_onPrice = pd.pivot_table(df, values='Price', index='Path', columns='is_peak_season', aggfunc='mean')
    print(Path_is_Peak_Session_onPrice)

    plt.figure(figsize=(10,6))
    sns.heatmap(
        Path_is_Peak_Session_onPrice,
        cmap='coolwarm'
    )
    plt.title("Path Price in Peak vs Non-Peak Season")
    plt.show()

    print("--------------------------")
    print('What is the impact of (airline + number of stops + flight duration) on the price?')
    impactOf_Air_Total_Dura_onPrice = df.groupby(['Airline','Total_Stops'])[['Duration_mins','Price']].mean().head()
    print(impactOf_Air_Total_Dura_onPrice)

    plt.figure(figsize=(10,6))
    sns.scatterplot(
        data=df,
        x='Duration_mins',
        y='Price',
        hue='Total_Stops',
        style='Airline',
        alpha=0.6
    )
    plt.title("Combined Effect of Airline, Stops & Duration on Price")
    plt.grid()
    plt.show()