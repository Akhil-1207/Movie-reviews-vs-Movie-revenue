import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

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
# CLEAN COLUMNS
# =========================
def clean_columns(df):
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

reviews = clean_columns(reviews)
sentiment = clean_columns(sentiment)
daywise = clean_columns(daywise)
weekwise = clean_columns(weekwise)

# =========================
# SMART COLUMN DETECTION
# =========================
def find_column(df, keyword):
    for col in df.columns:
        if keyword in col:
            return col
    return None

def find_numeric_column(df, keywords):
    for col in df.columns:
        if any(k in col for k in keywords) and "date" not in col:
            temp = pd.to_numeric(df[col], errors='coerce')
            if temp.notna().sum() > 0:
                return col
    return None

# Detect columns
review_date_col = find_column(reviews, "date")
sentiment_date_col = find_column(sentiment, "date")
daywise_date_col = find_column(daywise, "date")

rating_col = find_column(reviews, "rating")
platform_col = find_column(reviews, "platform")

collection_col = find_numeric_column(daywise, ["collection","revenue","gross","cr"])
sentiment_col = find_numeric_column(sentiment, ["sentiment","score","bert","polarity"])

# Safety check
if collection_col is None:
    st.error(f"❌ No numeric collection column found: {daywise.columns}")
    st.stop()

if sentiment_col is None:
    st.error(f"❌ No numeric sentiment column found: {sentiment.columns}")
    st.stop()

# =========================
# DATA CLEANING
# =========================
reviews[review_date_col] = pd.to_datetime(reviews[review_date_col], errors='coerce')
sentiment[sentiment_date_col] = pd.to_datetime(sentiment[sentiment_date_col], errors='coerce')
daywise[daywise_date_col] = pd.to_datetime(daywise[daywise_date_col], errors='coerce')

reviews[rating_col] = reviews[rating_col].fillna(0)

# Ensure numeric
daywise[collection_col] = pd.to_numeric(daywise[collection_col], errors='coerce')
sentiment[sentiment_col] = pd.to_numeric(sentiment[sentiment_col], errors='coerce')

# Drop invalid numeric rows
daywise = daywise.dropna(subset=[collection_col])
sentiment = sentiment.dropna(subset=[sentiment_col])

# =========================
# SAFE MERGE
# =========================
df_daywise = daywise.copy()
df_sentiment = sentiment.copy()

df_daywise[daywise_date_col] = pd.to_datetime(df_daywise[daywise_date_col], errors='coerce')
df_sentiment[sentiment_date_col] = pd.to_datetime(df_sentiment[sentiment_date_col], errors='coerce')

df_daywise = df_daywise.dropna(subset=[daywise_date_col])
df_sentiment = df_sentiment.dropna(subset=[sentiment_date_col])

df_daywise[daywise_date_col] = df_daywise[daywise_date_col].dt.date
df_sentiment[sentiment_date_col] = df_sentiment[sentiment_date_col].dt.date

df = pd.merge(
    df_daywise,
    df_sentiment,
    left_on=daywise_date_col,
    right_on=sentiment_date_col,
    how="left"
)

df[daywise_date_col] = pd.to_datetime(df[daywise_date_col])

# =========================
# FEATURES
# =========================
df = df.sort_values(daywise_date_col)
df['day_number'] = range(1, len(df) + 1)

df['week_number'] = "Week " + ((df['day_number'] - 1)//7 + 1).astype(str)
df['day_name'] = df[daywise_date_col].dt.day_name()
df['is_weekend'] = df['day_name'].isin(['Saturday','Sunday']).astype(int)

# SAFE tiers
df['collection_tier'] = np.where(df[collection_col] >= 20, "Blockbuster",
                        np.where(df[collection_col] >= 5, "Strong",
                        np.where(df[collection_col] >= 1, "Moderate", "Low")))

df['sentiment_label'] = np.where(df[sentiment_col] >= 0.6, "Positive",
                        np.where(df[sentiment_col] >= 0.4, "Neutral", "Negative"))

df['ma3_sentiment'] = df[sentiment_col].rolling(3).mean()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🎛 Filters")

date_range = st.sidebar.date_input(
    "Date Range",
    [df[daywise_date_col].min(), df[daywise_date_col].max()]
)

# =========================
# FILTER
# =========================
filtered = df[
    (df[daywise_date_col] >= pd.to_datetime(date_range[0])) &
    (df[daywise_date_col] <= pd.to_datetime(date_range[1]))
]

# =========================
# KPI
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Total Collection", round(filtered[collection_col].sum(),2))
col2.metric("Avg Sentiment", round(filtered[sentiment_col].mean(),3))
col3.metric("Total Records", len(filtered))

# =========================
# CHARTS
# =========================
st.plotly_chart(px.line(filtered, x=daywise_date_col, y=collection_col),
                use_container_width=True)

st.plotly_chart(px.scatter(filtered,
                           x=sentiment_col,
                           y=collection_col,
                           size=collection_col),
                use_container_width=True)

# =========================
# INSIGHTS
# =========================
corr = filtered[[collection_col, sentiment_col]].corr().iloc[0,1]

st.write(f"Correlation: {round(corr,3)}")

if corr > 0.6:
    st.success("Strong positive relationship")
elif corr > 0.3:
    st.warning("Moderate relationship")
else:
    st.error("Weak relationship")
