import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet

st.set_page_config(page_title="Olist Revenue Forecast", layout="wide")
st.title("Revenue Forecast (Olist)")

@st.cache_data
def load_data():
    orders = pd.read_csv("data/olist_orders_dataset.csv")
    items = pd.read_csv("data/olist_order_items_dataset.csv")

    items['revenue'] = items['price'] + items['freight_value']
    df = pd.merge(orders, items[['order_id', 'revenue']], on='order_id')
    df = df[df['order_status'] == 'delivered']
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    daily_rev = (
        df.groupby(df['order_purchase_timestamp'].dt.date)['revenue']
        .sum().reset_index().rename(columns={'order_purchase_timestamp': 'ds', 'revenue': 'y'})
    )
    daily_rev['ds'] = pd.to_datetime(daily_rev['ds'])
    return daily_rev

df = load_data()

# prophet mode
@st.cache_resource
def fit_prophet(df):
    holidays = pd.DataFrame({
        'holiday': 'brazil_holiday',
        'ds': pd.to_datetime([
            '2017-01-01', '2017-04-14', '2017-05-01', '2017-09-07',
            '2017-10-12', '2017-11-02', '2017-11-15', '2017-12-25'
        ]),
        'lower_window': 0,
        'upper_window': 1
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        holidays=holidays,
        seasonality_mode='multiplicative'
    )
    model.fit(df)
    return model

model = fit_prophet(df)

st.sidebar.title("Forecast Settings")
periods = st.sidebar.slider("Forecast Days", min_value=30, max_value=180, step=30, value=90)

future = model.make_future_dataframe(periods=periods)
forecast = model.predict(future)


st.subheader("Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)


st.subheader(f"Forecasted Revenue (Next {periods} Days)")
forecast_out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
st.dataframe(forecast_out.style.format({'yhat': '{:,.0f}', 'yhat_lower': '{:,.0f}', 'yhat_upper': '{:,.0f}'}))

predicted_rev = forecast_out['yhat'].sum()
lower = forecast_out['yhat_lower'].sum()
upper = forecast_out['yhat_upper'].sum()

st.markdown(f"""
### Summary
- **Total Forecasted Revenue**: BRL {predicted_rev:,.0f}
- **95% Confidence Range**: BRL {lower:,.0f} – BRL {upper:,.0f}
""")
