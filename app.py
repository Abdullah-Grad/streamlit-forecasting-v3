import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value

plt.style.use('fivethirtyeight')

st.set_page_config(layout='wide')
st.markdown(
    "<div style='text-align: center;'><img src='https://raw.githubusercontent.com/Abdullah-Grad/streamlit-forecasting-v2/main/logo.png' width='200'></div>",
    unsafe_allow_html=True
)
st.title("Demand Forecasting and Workforce Sizing 📈")

uploaded_file = st.file_uploader("📄 Upload your demand data (Excel format)", type=["xlsx"])

if uploaded_file:
    with st.spinner("🔄 Loading and preparing data..."):
        df = pd.read_excel(uploaded_file)
        month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df_long = df.melt(id_vars='Year', value_vars=month_cols,
                          var_name='Month', value_name='Demand')
        df_long['Date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month'], format='%Y-%b')
        df_long = df_long.sort_values('Date').reset_index(drop=True)
        df_long.set_index('Date', inplace=True)

    def add_promotion_factors(df):
        df['Promotion'] = 0
        for index, row in df.iterrows():
            if (row['ds'].month == 4 and row['ds'].year in [2023, 2024]) or \
               (row['ds'].month == 5 and row['ds'].year in [2020, 2021, 2022]) or \
               (row['ds'].month == 6 and row['ds'].year == 2019):
                df.at[index, 'Promotion'] = 1
            elif row['ds'].month in [9, 11, 12] or (row['ds'].month == 2 and row['ds'].year >= 2022):
                df.at[index, 'Promotion'] = 1
        return df

    with st.spinner("📊 Running simplified cross-validation..."):
        initial_window = 36
        train = df_long.iloc[:initial_window]
        test = df_long.iloc[initial_window:initial_window + 1]
        actual = test['Demand'].values[0] if not test.empty else 0

        try:
            sarima_model_cv = SARIMAX(train['Demand'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
            sarima_pred = sarima_model_cv.get_forecast(steps=1).predicted_mean.values[0]
        except:
            sarima_pred = 0

        df_prophet = train.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
        df_prophet['cap'] = df_prophet['y'].max() * 3
        df_prophet['floor'] = df_prophet['y'].min() * 0.5
        df_prophet['company_growth'] = df_prophet['ds'].dt.year - 2017
        df_prophet = add_promotion_factors(df_prophet)

        model_prophet_cv = Prophet(growth='logistic', yearly_seasonality=True,
                                   weekly_seasonality=False, daily_seasonality=False)
        model_prophet_cv.add_regressor('company_growth')
        model_prophet_cv.add_regressor('Promotion')
        model_prophet_cv.fit(df_prophet[['ds', 'y', 'cap', 'floor', 'company_growth', 'Promotion']])

        future = model_prophet_cv.make_future_dataframe(periods=1, freq='MS')
        future['cap'] = df_prophet['cap'].iloc[0]
        future['floor'] = df_prophet['floor'].iloc[0]
        future['company_growth'] = future['ds'].dt.year - 2017
        future = add_promotion_factors(future)
        prophet_pred = model_prophet_cv.predict(future)['yhat'].values[-1]

        try:
            hw_model_cv = ExponentialSmoothing(train['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit()
            hw_pred = hw_model_cv.forecast(1).values[0]
        except:
            hw_pred = train['Demand'].mean()

        best_mae = float('inf')
        best_weights = (1/3, 1/3, 1/3)
        for w1 in np.arange(0, 1.1, 0.1):
            for w2 in np.arange(0, 1.1 - w1, 0.1):
                w3 = 1 - w1 - w2
                blended = w1 * sarima_pred + w2 * prophet_pred + w3 * hw_pred
                mae = abs(actual - blended)
                if mae < best_mae:
                    best_mae = mae
                    best_weights = (w1, w2, w3)

        w1, w2, w3 = best_weights

    with st.spinner("📅 Forecasting and optimization..."):
        sarima_model = SARIMAX(df_long['Demand'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
        sarima_future = sarima_model.get_forecast(steps=12).predicted_mean
        future_index = pd.date_range(start=df_long.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
        sarima_future.index = future_index

        df_prophet_full = df_long.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
        df_prophet_full['cap'] = df_prophet_full['y'].max() * 3
        df_prophet_full['floor'] = df_prophet_full['y'].min() * 0.5
        df_prophet_full['company_growth'] = df_prophet_full['ds'].dt.year - 2017
        df_prophet_full = add_promotion_factors(df_prophet_full)

        model_prophet = Prophet(growth='logistic', yearly_seasonality=True,
                                weekly_seasonality=False, daily_seasonality=False)
        model_prophet.add_regressor('company_growth')
        model_prophet.add_regressor('Promotion')
        model_prophet.fit(df_prophet_full[['ds', 'y', 'cap', 'floor', 'company_growth', 'Promotion']])
        future = model_prophet.make_future_dataframe(periods=12, freq='MS')
        future['cap'] = df_prophet_full['cap'].iloc[0]
        future['floor'] = df_prophet_full['floor'].iloc[0]
        future['company_growth'] = future['ds'].dt.year - 2017
        future = add_promotion_factors(future)
        prophet_future = model_prophet.predict(future)['yhat'].values[-12:]

        hw_model_full = ExponentialSmoothing(df_long['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit()
        hw_future = hw_model_full.forecast(12).values

        combined_forecast = w1 * sarima_future.values + w2 * prophet_future + w3 * hw_future

        M, S = 12, 3
        Productivity = 23
        Cost = 8.5
        Days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        Hours = [6, 6, 6]
        model = LpProblem("Workforce_Scheduling", LpMinimize)
        X = {(i, j): LpVariable(f"X_{i}_{j}", lowBound=0, cat='Integer') for i in range(M) for j in range(S)}
        model += lpSum(Cost * X[i, j] * Hours[j] * Days[i] for i in range(M) for j in range(S))
        for i in range(M):
            model += lpSum(Productivity * X[i, j] * Hours[j] * Days[i] for j in range(S)) >= combined_forecast[i]
        model.solve()
        total_cost = value(model.objective)

    st.success(f"✅ Forecast Complete | SARIMA={w1:.2f}, Prophet={w2:.2f}, HW={w3:.2f} | Total Labor Cost: 💰 {total_cost:,.2f} SAR")

    results = [(future_index[i].strftime('%B'), combined_forecast[i], sum(value(X[i, j]) for j in range(S))) for i in range(M)]
    st.dataframe(pd.DataFrame(results, columns=["Month", "Forecasted Demand", "Workers Required"]))

    def plot_forecast(title, y_values):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df_long.index, df_long['Demand'], label='Historical Demand', marker='o')
        ax.plot(future_index, y_values, label=title, linestyle='--', marker='s')
        ax.set_title(f"{title} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Demand")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    st.subheader("📊 Individual Forecast Plots")
    plot_forecast("📈 SARIMA", sarima_future)
    plot_forecast("🔮 Prophet", prophet_future)
    plot_forecast("📉 Holt-Winters", hw_future)

    st.subheader("🧪 Blended Forecast")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_long.index, df_long['Demand'], label='Historical Demand', marker='o')
    ax.plot(future_index, combined_forecast, label='Blended Forecast', linestyle='--', marker='x')
    ax.set_title("Combined Weighted Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("📉 In-Sample Fit vs. Actual Demand")
    sarima_fitted = sarima_model.fittedvalues
    hw_fitted = hw_model_full.fittedvalues

    df_prophet_fit = df_long.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
    df_prophet_fit['cap'] = df_prophet_fit['y'].max() * 3
    df_prophet_fit['floor'] = df_prophet_fit['y'].min() * 0.5
    df_prophet_fit['company_growth'] = df_prophet_fit['ds'].dt.year - 2017
    df_prophet_fit = add_promotion_factors(df_prophet_fit)

    model_prophet_fit = Prophet(growth='logistic', yearly_seasonality=True)
    model_prophet_fit.add_regressor('company_growth')
    model_prophet_fit.add_regressor('Promotion')
    model_prophet_fit.fit(df_prophet_fit[['ds', 'y', 'cap', 'floor', 'company_growth', 'Promotion']])
    future_fit = df_prophet_fit[['ds', 'cap', 'floor', 'company_growth', 'Promotion']]
    prophet_fitted = model_prophet_fit.predict(future_fit)['yhat'].values

    combined_fitted = w1 * sarima_fitted.values + w2 * prophet_fitted + w3 * hw_fitted.values
    combined_fitted_series = pd.Series(combined_fitted, index=df_long.index[:len(combined_fitted)])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_long.index, df_long['Demand'], label='Actual Demand', marker='o')
    ax.plot(combined_fitted_series.index, combined_fitted_series, label='Fitted Forecast', linestyle='--', marker='x')
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.set_title("In-Sample Fit")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.subheader("🧮 Error Metrics")
    mae = mean_absolute_error(df_long['Demand'], combined_fitted_series)
    mape = mean_absolute_percentage_error(df_long['Demand'], combined_fitted_series) * 100
    st.write(f"**Cross-Validation MAE (1-Step)**: {best_mae:.2f}")
    st.write(f"**In-Sample MAE**: {mae:.2f}")
    st.write(f"**In-Sample MAPE**: {mape:.2f}%")
