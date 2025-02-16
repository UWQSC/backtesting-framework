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

#Determines if each company has only 1 row of data in df
company_counts = df.groupby('comp_name').size()
single_row_companies = company_counts[company_counts == 1].index
is_single_row = company_name in single_row_companies

yr_option = st.selectbox('Year: ', df_company['year'].unique(), index=None)

if yr_option is None:
    st.write("Graph for :" , option)
    #Not Given Year -> Closing and Opening Average Price (Line Graph)
    plt.figure(figsize=(15, 6))

    if is_single_row:
        #Plots scatterplot with singular point for proper readability
        plt.scatter(df_company['year'], df_company['prc'], color='blue', label="Price")
        plt.xticks([df_company['year'].values[0]]) # Displays singular year on x-axis
    else:
        #Plots line graph given that the user didn't select an option for the year
        sns.lineplot(data=df_company, x="year", y="prc")
        plt.xticks(df['year'].unique()) # Ensure all years are shown on the x-axis

    plt.title(f"Price vs. Year for {company_name}")
    plt.xlabel("Year")
    plt.ylabel("Price (prc)")
    st.pyplot(plt)

else:
    df_company_yr = df_company[df_company["year"] == yr_option].copy() # Creates dataframe for selected year of selected company

    st.write("Graph for :" , option)
    #Given Year -> Closing and Opening Average Price (Line Graph)
    plt.figure(figsize=(15, 6))

    if is_single_row:
        #Plots scatterplot with singular point for proper readability
        plt.scatter(df_company_yr['month'], df_company_yr['prc'], color='blue', label="Price")
        plt.xticks([df_company_yr['month'].values[0]]) # Displays singular month on x-axis
    else:
        #Plots line graph given that the user selected a year from dropdown
        sns.lineplot(data=df_company_yr, x="month", y="prc")
        plt.xticks(df_company_yr['month'].unique()) # Ensure all months of selected year are shown on the x-axis

    plt.title(f"Price vs. Month in {yr_option} for {company_name}")
    plt.xlabel("Month")
    plt.ylabel("Price (prc)")
    st.pyplot(plt)

#Create a average index such that simulates the trading 