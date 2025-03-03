import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Loading parquet data file (converted from CSV)
@st.cache_data
def load_data():
    return pd.read_parquet('data/hackathon_sample_v2.parquet')
df = load_data()

st.write("Top 1000 stocks on NASDAQ")

option = st.selectbox('Select a Company: ', df['comp_name'].unique())
'You selected: ' , option
company_name = option

df_company = df[df["comp_name"] == company_name].copy() # Creates dataframe for selected company
df_company['date'] = pd.to_datetime(df_company['date'].astype(str), format='%Y%m%d')

#Determines if each company has only 1 row of data in df
company_counts = df.groupby('comp_name').size()
single_row_companies = company_counts[company_counts == 1].index
is_single_row = company_name in single_row_companies

#Dropdowns for year and month selections
st.write("Select your desired month range:")
yr_option_start = st.selectbox('Start Year: ', df_company['year'].unique())
month_option_start = st.selectbox('Start Month: ', list(range(1, 13)))
yr_option_end = st.selectbox('End Year: ', df_company['year'].unique())
month_option_end = st.selectbox('End Month: ', list(range(1, 13)))

if yr_option_start is None or yr_option_start is None:
    st.write("Please choose a year for both the start and end")
elif month_option_start is None or month_option_end is None:
    st.write("Please choose a month for both the start and end")
elif (yr_option_start, month_option_start) > (yr_option_end, month_option_end):
    st.write("Please choose an earlier starting year/month or a later ending year/month")
elif yr_option_start == yr_option_end:
    #Creates dataframe for selected month range of selected year of selected company
    df_company_yr = df_company[(df_company["year"] == yr_option_start)].copy()
    df_company_mth = df_company_yr[(month_option_start <= df_company_yr["month"]) & (df_company_yr["month"] <= month_option_end)].copy()

    st.write("Graph for :" , option)
    #Given Same Year -> Closing and Opening Average Price (Line Graph)
    plt.figure(figsize=(15, 6))

    if df_company_mth.empty:
        st.write("No stock data available for selected month(s) for", company_name)
    elif month_option_start == month_option_end:
        #Plots scatterplot with singular point for proper readability
        plt.scatter(df_company_mth['month'], df_company_mth['prc'], color='blue', label="Price")
        plt.xticks([df_company_mth['month'].values[0]]) # Displays singular month on x-axis
        plt.title(f"Stock Price for Month {month_option_start} in {yr_option_start} for {company_name}") 
    elif is_single_row:
        #Plots scatterplot with singular point for proper readability
        plt.scatter(df_company_mth['month'], df_company_mth['prc'], color='blue', label="Price")
        plt.xticks([df_company_mth['month'].values[0]]) # Displays singular month on x-axis
        plt.title(f"Stock Price for Month {[df_company_mth['month'].values[0]]} in {yr_option_start} for {company_name}")
    else:
        #Plots line graph given that the user selected month range of a singular year from dropdown
        sns.lineplot(data=df_company_mth, x="month", y="prc")
        plt.xticks(df_company_mth['month'].unique()) # Ensure all selected months of selected year are shown on the x-axis
        plt.title(f"Stock Price Evolution Between Months {month_option_start} to {month_option_end} in {yr_option_start} for {company_name}")

    plt.xlabel("Month")
    plt.ylabel("Price (prc)")
    st.pyplot(plt)
else:
    df_company['date'] = pd.to_datetime(df_company['date'], format='%Y%m%d', errors='coerce')
    start_date = pd.Timestamp(year=yr_option_start, month=month_option_start, day=1)
    end_date = pd.Timestamp(year=yr_option_end, month=month_option_end, day=1) + pd.offsets.MonthEnd(0)

    #Filter DataFrame based on the converted 'date' column & selections for month and year range
    df_company_mth = df_company[(df_company['date'] >= start_date) & (df_company['date'] <= end_date)].copy()
    df_company_mth['month_year'] = df_company_mth['date'].dt.to_period('M').dt.to_timestamp()

    st.write("Graph for :" , option)
    #Given 2 Different Years with Months -> Closing and Opening Average Price (Line Graph)
    plt.figure(figsize=(15, 6))

    #Plots line graph given that the user selected an appropriate year and month range
    sns.lineplot(data=df_company_mth, x="month_year", y="prc")
    plt.xticks(rotation = 45)
    plt.title(f"Stock Price Evolution Between {yr_option_start} Month {month_option_start} and {yr_option_end} Month {month_option_end} for {company_name}")
    plt.xlabel("Month & Year")
    plt.ylabel("Price (prc)")
    st.pyplot(plt)

#Create a average index such that simulates the trading 