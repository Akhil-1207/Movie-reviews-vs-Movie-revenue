import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="🎬 Movie Dashboard", layout="wide")

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp {background: linear-gradient(to right, #141e30, #243b55); color: white;}
section[data-testid="stSidebar"] {background-color: #111;}
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Impact of Movie Reviews on Box Office")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    try:
        reviews = pd.read_csv(os.path.join(BASE_DIR, "Merged_War2_Cleaned_Reviews.csv"))
        sentiment = pd.read_csv(os.path.join(BASE_DIR, "War 2 sentiment scores.csv"))
        daywise = pd.read_csv(os.path.join(BASE_DIR, "War2_DayWise_Collections.csv"))
        weekwise = pd.read_csv(os.path.join(BASE_DIR, "War2_WeekWise_Collections.csv"))
    except Exception as e:
        st.error(f"❌ File loading error: {e}")
        st.stop()
    return reviews, sentiment, daywise, weekwise

reviews, sentiment, daywise, weekwise = load_data()

# =========================
# CLEAN COLUMN NAMES
# =========================
def clean_columns(df):
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

reviews = clean_columns(reviews)
sentiment = clean_columns(sentiment)
daywise = clean_columns(daywise)
weekwise = clean_columns(weekwise)

# =========================
# AUTO DETECT COLUMNS
# =========================
def find_column(df, keyword):
    for col in df.columns:
        if keyword in col:
            return col
    return None

def find_best_sentiment_column(df):
    keywords = ["sentiment", "score", "bert", "polarity"]
    for key in keywords:
        for col in df.columns:
            if key in col:
                return col
    return None

def find_best_collection_column(df):
    keywords = ["collection", "revenue", "gross", "cr"]
    for key in keywords:
        for col in df.columns:
            if key in col:
                return col
    return None

# Detect columns
review_date_col = find_column(reviews, "date")
sentiment_date_col = find_column(sentiment, "date")
daywise_date_col = find_column(daywise, "date")

rating_col = find_column(reviews, "rating")
platform_col = find_column(reviews, "platform")

collection_col = find_best_collection_column(daywise)
sentiment_col = find_best_sentiment_column(sentiment)

# =========================
# SAFETY CHECK (IMPORTANT)
# =========================
if sentiment_col is None:
    st.error(f"❌ Sentiment column not found. Columns: {list(sentiment.columns)}")
    st.stop()

if collection_col is None:
    st.error(f"❌ Collection column not found. Columns: {list(daywise.columns)}")
    st.stop()

# =========================
# DATA CLEANING
# =========================
reviews[review_date_col] = pd.to_datetime(reviews[review_date_col], errors='coerce')
sentiment[sentiment_date_col] = pd.to_datetime(sentiment[sentiment_date_col], errors='coerce')
daywise[daywise_date_col] = pd.to_datetime(daywise[daywise_date_col], errors='coerce')

reviews[rating_col] = reviews[rating_col].fillna(0)

daywise[collection_col] = pd.to_numeric(daywise[collection_col], errors='coerce').round(2)
sentiment[sentiment_col] = pd.to_numeric(sentiment[sentiment_col], errors='coerce').round(3)

# =========================
# MERGE
# =========================
df = pd.merge(
    daywise,
    sentiment,
    left_on=daywise_date_col,
    right_on=sentiment_date_col,
    how="left"
)

df = df.sort_values(daywise_date_col)
df['day_number'] = range(1, len(df) + 1)

df['week_number'] = "Week " + ((df['day_number'] - 1)//7 + 1).astype(str)
df['day_name'] = df[daywise_date_col].dt.day_name()
df['is_weekend'] = df['day_name'].isin(['Saturday','Sunday']).astype(int)

# =========================
# FEATURES
# =========================
df['collection_tier'] = np.select(
    [df[collection_col] >= 20,
     df[collection_col] >= 5,
     df[collection_col] >= 1],
    ['Blockbuster', 'Strong', 'Moderate'],
    default='Low'
)

df['sentiment_label'] = np.select(
    [df[sentiment_col] >= 0.6,
     df[sentiment_col] >= 0.4],
    ['Positive', 'Neutral'],
    default='Negative'
)

df['ma3_sentiment'] = df[sentiment_col].rolling(3).mean()

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🎛 Filters")

date_range = st.sidebar.date_input(
    "📅 Date Range",
    [df[daywise_date_col].min(), df[daywise_date_col].max()]
)

day_filter = st.sidebar.multiselect(
    "📆 Day",
    df['day_name'].unique(),
    default=df['day_name'].unique()
)

week_filter = st.sidebar.multiselect(
    "📊 Week",
    df['week_number'].unique(),
    default=df['week_number'].unique()
)

source_filter = st.sidebar.multiselect(
    "🗣 Review Source",
    reviews[platform_col].dropna().unique(),
    default=reviews[platform_col].dropna().unique()
)

# =========================
# APPLY FILTERS
# =========================
filtered = df[
    (df[daywise_date_col] >= pd.to_datetime(date_range[0])) &
    (df[daywise_date_col] <= pd.to_datetime(date_range[1])) &
    (df['day_name'].isin(day_filter)) &
    (df['week_number'].isin(week_filter))
]

filtered_reviews = reviews[
    reviews[platform_col].isin(source_filter)
]

# =========================
# KPI
# =========================
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Collection", round(filtered[collection_col].sum(),2))
col2.metric("😊 Avg Sentiment", round(filtered[sentiment_col].mean(),3))
col3.metric("🗣 Reviews", len(filtered_reviews))
col4.metric("🔥 Peak Day", round(filtered[collection_col].max(),2))

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🧠 Sentiment", "💰 Revenue"])

with tab1:
    fig1 = px.bar(filtered, x=daywise_date_col, y=collection_col,
                  color="collection_tier", title="Daily Collection")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(filtered, x=sentiment_col, y=collection_col,
                      color="collection_tier", size=collection_col,
                      title="Sentiment vs Collection")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    fig3 = px.line(filtered, x=daywise_date_col,
                   y=[sentiment_col, "ma3_sentiment"],
                   title="Sentiment Trend")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(filtered_reviews, x=rating_col, nbins=10,
                        title="Ratings Distribution")
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    week_col = find_best_collection_column(weekwise)
    week_name_col = find_column(weekwise, "week")

    fig5 = px.bar(weekwise, x=week_name_col, y=week_col,
                  title="Weekly Revenue")
    st.plotly_chart(fig5, use_container_width=True)

# =========================
# INSIGHTS
# =========================
st.subheader("📌 Insights")

corr = filtered[[collection_col, sentiment_col]].corr().iloc[0,1]

st.write(f"Correlation: **{round(corr,3)}**")

if corr > 0.6:
    st.success("🔥 Strong positive relationship")
elif corr > 0.3:
    st.warning("⚠️ Moderate relationship")
else:
    st.error("❌ Weak relationship")
