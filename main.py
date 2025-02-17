import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/hackathon_sample_v2.csv')

st.write("Top 1000 stocks on NASDAQ")



#company_names = df['Company'].unique()

option = st.selectbox('Select a Company: ', df['comp_name'].unique())
'You selected: ' , option
company_name = option

df_company = df[df["comp_name"] == company_name].copy() # Creates dataframe for selected company
df_company['date'] = pd.to_datetime(df_company['date'].astype(str), format='%Y%m%d')

#Determines if each company has only 1 row of data in df
company_counts = df.groupby('comp_name').size()
single_row_companies = company_counts[company_counts == 1].index
is_single_row = company_name in single_row_companies

st.write("Select your date range:")
yr_option_start = st.selectbox('Start: ', df_company['year'].unique())
yr_option_end = st.selectbox('End: ', df_company['year'].unique())

if yr_option_start is None or yr_option_start is None:
    st.write("Please choose a year for both the start and end")
elif yr_option_start > yr_option_end:
    st.write("Please choose an earlier year or later year")
elif yr_option_start == yr_option_end:
    df_company_yr = df_company[df_company["year"] == yr_option_start].copy() # Creates dataframe for selected year of selected company

    st.write("Graph for :" , option)
    #Given Same Year -> Closing and Opening Average Price (Line Graph)
    plt.figure(figsize=(15, 6))

    if is_single_row:
        #Plots scatterplot with singular point for proper readability
        plt.scatter(df_company_yr['month'], df_company_yr['prc'], color='blue', label="Price")
        plt.xticks([df_company_yr['month'].values[0]]) # Displays singular month on x-axis
    else:
        #Plots line graph given that the user selected a year from dropdown
        sns.lineplot(data=df_company_yr, x="month", y="prc")
        plt.xticks(df_company_yr['month'].unique()) # Ensure all months of selected year are shown on the x-axis

    plt.title(f"Price vs. Month in {yr_option_start} for {company_name}")
    plt.xlabel("Month")
    plt.ylabel("Price (prc)")
    st.pyplot(plt)
else:
    # Creates dataframe for selected year range of selected company
    df_company_yr = df_company[(yr_option_start <= df_company["year"]) & (df_company["year"] <= yr_option_end)].copy()
    df_company_yr['month_year'] = df_company_yr['date'].dt.to_period('M').dt.to_timestamp()

    st.write("Graph for :" , option)
    #Given 2 Different Years -> Closing and Opening Average Price (Line Graph)
    plt.figure(figsize=(15, 6))

    #Plots line graph given that the user selected an appropriate year range
    sns.lineplot(data=df_company_yr, x="month_year", y="prc")
    plt.xticks(rotation = 45)
    plt.title(f"Price vs. Year Between {yr_option_start} and {yr_option_end} for {company_name}")
    plt.xlabel("Month & Year")
    plt.ylabel("Price (prc)")
    st.pyplot(plt)

#Create a average index such that simulates the trading 