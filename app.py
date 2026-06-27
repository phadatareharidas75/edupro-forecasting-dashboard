import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="EduPro Forecasting Dashboard", layout="wide")

st.title("📊 EduPro: Course Demand & Revenue Forecasting")
st.markdown("---")

@st.cache_data
def load_and_prepare_data():
    transactions_df = pd.read_excel("EduPro Online Platform.xlsx", sheet_name="Transactions")
    courses_df = pd.read_excel("EduPro Online Platform.xlsx", sheet_name="Courses")
    
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    merged_df = pd.merge(transactions_df, courses_df, on="CourseID", how="left")
    
    daily_df = merged_df.groupby(['TransactionDate', 'CourseCategory']).agg(
        Enrollments=('TransactionID', 'count'),
        Revenue=('Amount', 'sum')
    ).reset_index()
    
    return daily_df

try:
    df = load_and_prepare_data()

    st.sidebar.header("⚙️ Filters")
    categories = ['All'] + list(df['CourseCategory'].unique())
    selected_category = st.sidebar.selectbox("Select Course Category:", categories)

    if selected_category != 'All':
        plot_df = df[df['CourseCategory'] == selected_category]
    else:
        plot_df = df.groupby('TransactionDate').sum(numeric_only=True).reset_index()

    plot_df = plot_df.sort_values('TransactionDate')

    plot_df['Date_Ordinal'] = plot_df['TransactionDate'].map(pd.Timestamp.toordinal)

    X = plot_df[['Date_Ordinal']]
    y_demand = plot_df['Enrollments']
    y_revenue = plot_df['Revenue']

    model_demand = LinearRegression().fit(X, y_demand)
    model_revenue = LinearRegression().fit(X, y_revenue)

    future_dates = pd.date_range(start=plot_df['TransactionDate'].max() + pd.Timedelta(days=1), periods=30, freq='D')
    future_df = pd.DataFrame({'TransactionDate': future_dates})
    future_df['Date_Ordinal'] = future_df['TransactionDate'].map(pd.Timestamp.toordinal)

    # इथे आपण एरर फिक्स करण्यासाठी np.clip() चा वापर केला आहे जेणेकरून निगेटिव्ह व्हॅल्यू येणार नाहीत
    demand_preds = model_demand.predict(future_df[['Date_Ordinal']])
    revenue_preds = model_revenue.predict(future_df[['Date_Ordinal']])
    
    future_df['Predicted_Enrollments'] = np.clip(demand_preds, 0, None).astype(int)
    future_df['Predicted_Revenue'] = np.clip(revenue_preds, 0.0, None).astype(float)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🔥 Expected Demand ({selected_category}) - Next 30 Days")
        fig_demand = px.line(future_df, x='TransactionDate', y='Predicted_Enrollments', 
                             title="Demand Forecast Trend", labels={'Predicted_Enrollments': 'Enrollments', 'TransactionDate': 'Date'})
        fig_demand.update_traces(line_color='#FF4B4B', line_width=3)
        st.plotly_chart(fig_demand, use_container_width=True)

    with col2:
        st.subheader(f"💰 Expected Revenue ({selected_category}) - Next 30 Days")
        fig_rev = px.line(future_df, x='TransactionDate', y='Predicted_Revenue', 
                          title="Revenue Forecast Trend", labels={'Predicted_Revenue': 'Revenue (₹)', 'TransactionDate': 'Date'})
        fig_rev.update_traces(line_color='#00CC96', line_width=3)
        st.plotly_chart(fig_rev, use_container_width=True)

    st.subheader("📋 Prediction Data Sheet (Next 30 Days)")
    future_df_display = future_df[['TransactionDate', 'Predicted_Enrollments', 'Predicted_Revenue']].copy()
    future_df_display['Predicted_Revenue'] = future_df_display['Predicted_Revenue'].map('₹{:,.2f}'.format)
    st.dataframe(future_df_display.head(10), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Make sure 'EduPro Online Platform.xlsx' is in the same folder.")