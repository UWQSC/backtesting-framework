import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sma_algorithm import SimpleMovingAverageImpl


#Loading parquet data file (converted from CSV)
@st.cache_data
def load_data():
    return pd.read_parquet('data/hackathon_sample_v2.parquet')
df = load_data()

st.write("Top 1000 stocks on NASDAQ")

company_name = st.selectbox('Select a Company: ', df['comp_name'].unique())
'You selected: ' , company_name

df_company = df[df["comp_name"] == company_name].copy() # Creates dataframe for selected company
df_company['date'] = pd.to_datetime(df_company['date'].astype(str), format='%Y%m%d')

#Dropdowns for year and month selections
st.write("Select your desired month range:")
yr_option_start = st.selectbox('Start Year: ', df_company['year'].unique())
month_option_start = st.selectbox('Start Month: ', list(range(1, 13)))
yr_option_end = st.selectbox('End Year: ', df_company['year'].unique())
month_option_end = st.selectbox('End Month: ', list(range(1, 13)))

#Conditional checks based on time period selection
if yr_option_start is None or yr_option_end is None:
    st.warning("Please choose a year for both the start and end")
    st.stop()
elif month_option_start is None or month_option_end is None:
    st.warning("Please choose a month for both the start and end")
    st.stop()
elif (yr_option_start, month_option_start) > (yr_option_end, month_option_end):
    st.warning("Please choose an earlier starting year/month or a later ending year/month")
    st.stop()

#Creating filtered dataframe to work with
elif yr_option_start == yr_option_end:
    df_company_mth = df_company[
        (df_company["year"] == yr_option_start) & 
        (df_company["month"] >= month_option_start) & 
        (df_company["month"] <= month_option_end)
    ].copy()
else:
    df_company_mth = df_company[
        (
            ((df_company["year"] == yr_option_start) & (df_company["month"] >= month_option_start)) |
            ((df_company["year"] > yr_option_start) & (df_company["year"] < yr_option_end)) |
            ((df_company["year"] == yr_option_end) & (df_company["month"] <= month_option_end))
        )
    ].copy()

df_company_mth["month_year"] = df_company_mth["date"].dt.to_period("M").dt.to_timestamp()

# tab1, tab2 = st.tabs(["Stock Price Evolution", "Portfolio Simulation"])
st.write("Portfolio Simulation with Your Own Custom Trading Strategy!")
if df_company_mth.empty:
    st.warning("No stock data available for the selected date range.")
else:
    strat_list = ["SMA", "Custom"]
    strat = st.selectbox('Select your strategy for backtesting: ', strat_list)

    if strat == "SMA":
        df_sma = df[["stock_ticker", "date", "year", "month", "prc"]].copy()
        
        #Dropdown for selecting the stock ticker
        available_tickers = sorted(df_sma["stock_ticker"].dropna().unique())
        selected_tickers = st.multiselect("Select Tickers for Testing:", available_tickers)

        df_sma = df_sma.loc[df_sma["stock_ticker"].isin(selected_tickers)]
        df_sma['date'] = pd.to_datetime(df_sma['date'].astype(str), format='%Y%m%d')

        #Dropdowns for year and month selections
        st.write("Select your desired timeframe:")
        start_yr = st.selectbox('Starting Year: ', df_sma['year'].unique())
        start_mth = st.selectbox('Starting Month: ', list(range(1, 13)))
        end_yr = st.selectbox('Ending Year: ', df_sma['year'].unique())
        end_mth = st.selectbox('Ending Month: ', list(range(1, 13)))

        # Conditional checks for selected timeframe
        if start_yr is None or end_yr is None:
            st.warning("Please choose a year for both the start and end")
            st.stop()
        elif start_mth is None or end_mth is None:
            st.warning("Please choose a month for both the start and end")
            st.stop()
        elif (start_yr, start_mth) > (end_yr, end_mth):
            st.warning("Please choose an earlier starting year/month or a later ending year/month")
            st.stop()

        #Creating filtered dataframe to work with
        elif start_yr == end_yr:
            df_sma = df_sma[
                (df_sma["year"] == start_yr) & 
                (df_sma["month"] >= start_mth) & 
                (df_sma["month"] <= end_mth)
            ]
        else:
            df_sma = df_sma[
                (
                    ((df_sma["year"] == start_yr) & (df_sma["month"] >= start_mth)) |
                    ((df_sma["year"] > start_yr) & (df_sma["year"] < end_yr)) |
                    ((df_sma["year"] == end_yr) & (df_sma["month"] <= end_mth))
                )
            ]

        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=1_000_000, step=1000)

        #User selects SMA parameters
        short_window = st.slider("SMA Short Window (Months)", min_value=1, max_value=12, value=3)
        long_window = st.slider("SMA Long Window (Months)", min_value=3, max_value=24, value=6)

        #Run SMA backtest
        if st.button("Run SMA Backtest") and selected_tickers:
            st.write(f"Running SMA Strategy for: {', '.join(selected_tickers)}")

            sma_algo = SimpleMovingAverageImpl(
                tickers=selected_tickers,
                data=df_sma,
                parameters={"position_size": 0.1, "short_window": short_window, "long_window": long_window}
            )

            sma_portfolio = sma_algo.execute_trades(capital=initial_capital)
            metrics = sma_algo.calculate_metrics(sma_portfolio)

            st.subheader("SMA Portfolio Performance Metrics")
            for key, value in metrics.items():
                st.write(f"**{key}:** {value}")

            st.subheader("Portfolio Value Over Time")
            st.line_chart(sma_portfolio["capital"])

    else:
        st.write("Your current company selection is ", company_name)
        st.write("Please change above selections if you would like to change company & time period")

        #Shortlist of indicators that work with given monthly data
        valid_indicators = [
            "ret_1_0", "ret_3_1", "ret_6_1", "ret_12_1", "stock_exret", "prc", "market_equity", "eps_actual", "be_me", "sale_bev", "ocf_me",
            "sale_gr1", "at_gr1", "betadown_252d", "corr_1260d", "ivol_capm_21d"
        ]
        
        #Selection of entry & exit indicators and choice for AND or OR operator for indicators
        entry_indicators = st.multiselect("Select Entry Indicators (Buy Signal):", valid_indicators)
        entry_condition_type = st.radio("Combine Entry Conditions With:", ["AND", "OR"])
        
        exit_indicators = st.multiselect("Select Exit Indicators (Sell Signal):", valid_indicators)
        exit_condition_type = st.radio("Combine Exit Conditions With:", ["AND", "OR"])
        
        #Set threshold values for each selected condition
        indicator_thresholds = {}

        for ind in entry_indicators:
            #In both entry & exit
            if ind in exit_indicators:
                indicator_thresholds[f"entry_{ind}"] = st.number_input(
                    f"Entry Threshold for {ind}:", min_value=-1000000000.0, max_value=1000000000.0, value=0.0
                )
                indicator_thresholds[f"exit_{ind}"] = st.number_input(
                    f"Exit Threshold for {ind}:", min_value=-1000000000.0, max_value=1000000000.0, value=0.0
                )

            #Only in entry
            else:
                indicator_thresholds[f"entry_{ind}"] = st.number_input(
                    f"Entry Threshold for {ind}:", min_value=-1000000000.0, max_value=1000000000.0, value=0.0
                )

        for ind in exit_indicators:
            #Only in exit
            if ind not in entry_indicators:
                indicator_thresholds[f"exit_{ind}"] = st.number_input(
                    f"Exit Threshold for {ind}:", min_value=-1000000000.0, max_value=1000000000.0, value=0.0
                )

        #Portfolio simulation parameters
        initial_capital = 1_000_000
        capital = initial_capital
        holdings = 0

        #Handles missing values in filtered dataset using forward & backward fill
        #Also fills remaining missing values with 0
        df_company_mth.ffill(inplace=True)
        df_company_mth.bfill(inplace=True)
        df_company_mth.fillna(0, inplace=True)

        df_company_mth["portfolio_value"] = np.nan

        #Apply entry conditions
        if entry_indicators:
            entry_conditions = []
        
            for ind in entry_indicators:
                if ind in df_company_mth.columns:
                    threshold_key = f"entry_{ind}"
                    if threshold_key in indicator_thresholds:
                        entry_conditions.append(df_company_mth[ind] > indicator_thresholds[threshold_key])
            
            if entry_condition_type == "AND":
                df_company_mth["entry_signal"] = np.logical_and.reduce(entry_conditions) if entry_conditions else False
            else:
                df_company_mth["entry_signal"] = np.logical_or.reduce(entry_conditions) if entry_conditions else False
            
        else:
            df_company_mth["entry_signal"] = False
        
        #Apply exit conditions
        if exit_indicators:
            exit_conditions = []

            for ind in exit_indicators:
                if ind in df_company_mth.columns:
                    threshold_key = f"exit_{ind}"
                    if threshold_key in indicator_thresholds:
                        exit_conditions.append(df_company_mth[ind] < indicator_thresholds[threshold_key])
            
            if exit_condition_type == "AND":
                df_company_mth["exit_signal"] = np.logical_and.reduce(exit_conditions) if exit_conditions else False
            else:
                df_company_mth["exit_signal"] = np.logical_or.reduce(exit_conditions) if exit_conditions else False
        
        else:
            df_company_mth["exit_signal"] = False

        #Portfolio simulation
        for index, row in df_company_mth.iterrows():
            if row["entry_signal"]:
                #Determine number of shares to buy
                shares = capital // row["prc"]
                
                if shares > 0:
                    #Makes adjustment to current holdings & capital
                    holdings += shares
                    capital -= shares * row["prc"]
                
                #Updates portfolio value after an entry signal is met
                df_company_mth.at[index, "portfolio_value"] = capital + (holdings * row["prc"])

            elif row["exit_signal"] and holdings > 0:
                #Sells all shares    
                capital += holdings * row["prc"]
                holdings = 0
            
                #Updates portfolio value after an exit signal is met
                df_company_mth.at[index, "portfolio_value"] = capital
        
            else:
                #Default update for portfolio value
                df_company_mth.at[index, "portfolio_value"] = capital + (holdings * row["prc"])
        
        #Portfolio performance visualization
        st.write("Portfolio Performance Over Time")
        plt.figure(figsize=(12, 6))
        plt.plot(df_company_mth["month_year"], df_company_mth["portfolio_value"], label="Portfolio Value", color="green")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.title(f"Portfolio Simulation for {company_name}")
        plt.legend()
        st.pyplot(plt)
