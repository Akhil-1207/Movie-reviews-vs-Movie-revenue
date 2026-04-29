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
# CUSTOM STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #111;
}
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Impact of Movie Reviews on Box Office")

# =========================
# LOAD DATA (FIXED)
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
reviews.columns = reviews.columns.str.strip().str.replace(" ", "_")
sentiment.columns = sentiment.columns.str.strip().str.replace(" ", "_")
daywise.columns = daywise.columns.str.strip().str.replace(" ", "_")
weekwise.columns = weekwise.columns.str.strip().str.replace(" ", "_")

# =========================
# DATA CLEANING
# =========================
reviews['Review_Date'] = pd.to_datetime(reviews['Review_Date'], errors='coerce')
sentiment['Collection_Date'] = pd.to_datetime(sentiment['Collection_Date'], errors='coerce')
daywise['Collection_Date'] = pd.to_datetime(daywise['Collection_Date'], errors='coerce')

reviews['User_Rating'] = reviews['User_Rating'].fillna(0)

daywise['Daily_Collection_Cr'] = daywise['Daily_Collection_Cr'].round(2)
sentiment['Avg_BERT_Sentiment'] = sentiment['Avg_BERT_Sentiment'].round(3)

# =========================
# MERGE + FEATURES
# =========================
df = pd.merge(daywise, sentiment, on="Collection_Date", how="left")

df = df.sort_values("Collection_Date")
df['Day_Number'] = range(1, len(df) + 1)

df['Week_Number'] = "Week " + ((df['Day_Number'] - 1)//7 + 1).astype(str)
df['Day_Name'] = df['Collection_Date'].dt.day_name()
df['Is_Weekend'] = df['Day_Name'].isin(['Saturday','Sunday']).astype(int)

# Collection Tier
df['Collection_Tier'] = np.select(
    [df['Daily_Collection_Cr'] >= 20,
     df['Daily_Collection_Cr'] >= 5,
     df['Daily_Collection_Cr'] >= 1],
    ['Blockbuster', 'Strong', 'Moderate'],
    default='Low'
)

# Sentiment Label
df['Sentiment_Label'] = np.select(
    [df['Avg_BERT_Sentiment'] >= 0.6,
     df['Avg_BERT_Sentiment'] >= 0.4],
    ['Positive', 'Neutral'],
    default='Negative'
)

# Moving Average
df['MA3_Sentiment'] = df['Avg_BERT_Sentiment'].rolling(3).mean()

# =========================
# SIDEBAR FILTERS (SLICERS)
# =========================
st.sidebar.header("🎛 Filters")

date_range = st.sidebar.date_input(
    "📅 Date Range",
    [df['Collection_Date'].min(), df['Collection_Date'].max()]
)

day_filter = st.sidebar.multiselect(
    "📆 Day",
    df['Day_Name'].unique(),
    default=df['Day_Name'].unique()
)

week_filter = st.sidebar.multiselect(
    "📊 Week",
    df['Week_Number'].unique(),
    default=df['Week_Number'].unique()
)

source_filter = st.sidebar.multiselect(
    "🗣 Review Source",
    reviews['Review_Platform'].dropna().unique(),
    default=reviews['Review_Platform'].dropna().unique()
)

# Apply filters
filtered = df[
    (df['Collection_Date'] >= pd.to_datetime(date_range[0])) &
    (df['Collection_Date'] <= pd.to_datetime(date_range[1])) &
    (df['Day_Name'].isin(day_filter)) &
    (df['Week_Number'].isin(week_filter))
]

filtered_reviews = reviews[
    reviews['Review_Platform'].isin(source_filter)
]

# =========================
# KPI SECTION
# =========================
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Collection", round(filtered['Daily_Collection_Cr'].sum(),2))
col2.metric("😊 Avg Sentiment", round(filtered['Avg_BERT_Sentiment'].mean(),3))
col3.metric("🗣 Reviews", len(filtered_reviews))
col4.metric("🔥 Peak Day", round(filtered['Daily_Collection_Cr'].max(),2))

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🧠 Sentiment", "💰 Revenue"])

# =========================
# OVERVIEW
# =========================
with tab1:
    fig1 = px.bar(filtered,
        x="Collection_Date",
        y="Daily_Collection_Cr",
        color="Collection_Tier",
        title="Daily Collection Trend")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(filtered,
        x="Avg_BERT_Sentiment",
        y="Daily_Collection_Cr",
        color="Collection_Tier",
        size="Daily_Collection_Cr",
        title="Sentiment vs Collection")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# SENTIMENT
# =========================
with tab2:
    fig3 = px.line(filtered,
        x="Collection_Date",
        y=["Avg_BERT_Sentiment","MA3_Sentiment"],
        title="Sentiment Trend")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(filtered_reviews,
        x="User_Rating",
        nbins=10,
        title="Rating Distribution")
    st.plotly_chart(fig4, use_container_width=True)

# =========================
# REVENUE
# =========================
with tab3:
    fig5 = px.bar(weekwise,
        x="Week_Number",
        y="Weekly_Collection_Cr",
        title="Weekly Revenue")
    st.plotly_chart(fig5, use_container_width=True)

    weekend = filtered[filtered['Is_Weekend']==1]['Daily_Collection_Cr'].mean()
    weekday = filtered[filtered['Is_Weekend']==0]['Daily_Collection_Cr'].mean()

    comp = pd.DataFrame({
        "Type":["Weekend","Weekday"],
        "Collection":[weekend,weekday]
    })

    fig6 = px.bar(comp,
        x="Type",
        y="Collection",
        title="Weekend vs Weekday")
    st.plotly_chart(fig6, use_container_width=True)

# =========================
# INSIGHTS
# =========================
st.subheader("📌 Insights")

corr = filtered[['Daily_Collection_Cr','Avg_BERT_Sentiment']].corr().iloc[0,1]

st.write(f"Correlation: **{round(corr,3)}**")

if corr > 0.6:
    st.success("🔥 Strong positive relationship: sentiment drives revenue")
elif corr > 0.3:
    st.warning("⚠️ Moderate relationship observed")
else:
    st.error("❌ Weak relationship")

st.markdown("""
### 🔍 Key Insights
- Opening week contributes highest revenue  
- Positive sentiment aligns with peak collections  
- Weekend collections outperform weekdays  
- Revenue declines after early weeks  
""")
