import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/new_sample_data.csv')

st.write("Top 1000 stocks on NASDAQ")



#company_names = df['Company'].unique()

option = st.selectbox('Select a Company: ', df['comp_name'].unique())
'You selected: ' , option

st.write("Graph for :" , option)
#Given Month and Year -> Closing and Opening Average Price (Line Graph)
company_name = option
df_company = df[df["comp_name"] == company_name].copy()
plt.figure(figsize=(15, 6))
sns.lineplot(data=df_company, x="year", y="prc")
plt.title(f"Price vs. Year for {company_name}")
plt.xlabel("Year")
plt.ylabel("Price (prc)")
plt.xticks(df['year'].unique()) # Ensure all years are shown on the x-axis
st.pyplot(plt)

#Create a average index such that simulates the trading 
