import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Movie Dashboard", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(to right, #141e30, #243b55); color: white;}
section[data-testid="stSidebar"] {background-color: #111;}
[data-testid="metric-container"] {background-color: rgba(255,255,255,0.08); border-radius: 12px; padding: 15px;}
</style>
""", unsafe_allow_html=True)

st.title("Impact of Movie Reviews on Box Office")

@st.cache_data
def load_data():
    reviews = pd.read_csv("data/reviews.csv")
    sentiment = pd.read_csv("data/sentiment.csv")
    daywise = pd.read_csv("data/daywise.csv")
    weekwise = pd.read_csv("data/weekwise.csv")
    return reviews, sentiment, daywise, weekwise

reviews, sentiment, daywise, weekwise = load_data()

reviews['Review_Date'] = pd.to_datetime(reviews['Review_Date'], errors='coerce')
sentiment['Collection_Date'] = pd.to_datetime(sentiment['Collection_Date'], errors='coerce')
daywise['Collection_Date'] = pd.to_datetime(daywise['Collection_Date'], errors='coerce')

reviews['User_Rating'] = reviews['User_Rating'].fillna(0)

daywise['Daily_Collection_Cr'] = daywise['Daily_Collection_Cr'].round(2)
sentiment['Avg_BERT_Sentiment'] = sentiment['Avg_BERT_Sentiment'].round(3)

df = pd.merge(daywise, sentiment, on="Collection_Date", how="left")

df = df.sort_values("Collection_Date")
df['Day_Number'] = range(1, len(df) + 1)
df['Week_Number'] = "Week " + ((df['Day_Number'] - 1)//7 + 1).astype(str)
df['Day_Name'] = df['Collection_Date'].dt.day_name()
df['Is_Weekend'] = df['Day_Name'].isin(['Saturday','Sunday']).astype(int)

st.sidebar.header("Filters")

date_range = st.sidebar.date_input("Date Range",
    [df['Collection_Date'].min(), df['Collection_Date'].max()])

filtered = df[
    (df['Collection_Date'] >= pd.to_datetime(date_range[0])) &
    (df['Collection_Date'] <= pd.to_datetime(date_range[1]))
]

col1, col2, col3 = st.columns(3)
col1.metric("Total Collection", round(filtered['Daily_Collection_Cr'].sum(),2))
col2.metric("Avg Sentiment", round(filtered['Avg_BERT_Sentiment'].mean(),3))
col3.metric("Total Reviews", len(reviews))

fig = px.line(filtered, x="Collection_Date", y="Daily_Collection_Cr")
st.plotly_chart(fig, use_container_width=True)
