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

st.title("🎬 Movie Reviews vs Box Office Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    reviews = pd.read_csv(os.path.join(base, "Merged_War2_Cleaned_Reviews.csv"))
    sentiment = pd.read_csv(os.path.join(base, "War 2 sentiment scores.csv"))
    daywise = pd.read_csv(os.path.join(base, "War2_DayWise_Collections.csv"))
    weekwise = pd.read_csv(os.path.join(base, "War2_WeekWise_Collections.csv"))
    return reviews, sentiment, daywise, weekwise

reviews, sentiment, daywise, weekwise = load_data()

# =========================
# CLEAN COLUMN NAMES
# =========================
def clean(df):
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

reviews = clean(reviews)
sentiment = clean(sentiment)
daywise = clean(daywise)
weekwise = clean(weekwise)

# =========================
# AUTO DETECT IMPORTANT COLUMNS
# =========================
def find_col(df, key):
    for c in df.columns:
        if key in c:
            return c
    return None

def find_numeric(df, keys):
    for c in df.columns:
        if any(k in c for k in keys) and "date" not in c:
            val = pd.to_numeric(df[c], errors='coerce')
            if val.notna().sum() > 0:
                return c
    return None

# 🔥 FIX: Better platform detection
def find_platform_column(df):
    keywords = ["platform","source","site","channel"]
    for col in df.columns:
        if any(k in col for k in keywords):
            return col
    return None

date_day = find_col(daywise, "date")
date_sent = find_col(sentiment, "date")
date_review = find_col(reviews, "date")

collection = find_numeric(daywise, ["collection","cr","gross"])
sent_score = find_numeric(sentiment, ["sentiment","score","bert"])
rating = find_col(reviews, "rating")
platform = find_platform_column(reviews)

# 🔥 FIX: Safe fallback if platform not found
if platform is None:
    st.warning("⚠️ No review source column found → using default")
    reviews["review_source"] = "All"
    platform = "review_source"

# =========================
# CLEAN DATA
# =========================
daywise[date_day] = pd.to_datetime(daywise[date_day], errors='coerce')
sentiment[date_sent] = pd.to_datetime(sentiment[date_sent], errors='coerce')
reviews[date_review] = pd.to_datetime(reviews[date_review], errors='coerce')

daywise[collection] = pd.to_numeric(daywise[collection], errors='coerce')
sentiment[sent_score] = pd.to_numeric(sentiment[sent_score], errors='coerce')

# =========================
# MERGE
# =========================
daywise = daywise.dropna(subset=[date_day, collection])
sentiment = sentiment.dropna(subset=[date_sent, sent_score])

daywise[date_day] = daywise[date_day].dt.date
sentiment[date_sent] = sentiment[date_sent].dt.date

df = pd.merge(daywise, sentiment, left_on=date_day, right_on=date_sent, how="left")
df[date_day] = pd.to_datetime(df[date_day])

# =========================
# USER FRIENDLY COLUMNS
# =========================
df.rename(columns={
    date_day: "Date",
    collection: "Daily Collection (Cr)",
    sent_score: "Sentiment Score"
}, inplace=True)

# =========================
# FEATURES
# =========================
df = df.sort_values("Date")
df["Day"] = df["Date"].dt.day_name()
df["Week"] = "Week " + ((np.arange(len(df)) // 7) + 1).astype(str)

df["Collection Category"] = np.where(df["Daily Collection (Cr)"] > 20, "Blockbuster",
                             np.where(df["Daily Collection (Cr)"] > 5, "Strong",
                             np.where(df["Daily Collection (Cr)"] > 1, "Average", "Low")))

# =========================
# 🔥 NEW: BUTTON FILTERS (ADDED)
# =========================
st.subheader("⚡ Quick Filters")

colf1, colf2 = st.columns(2)

with colf1:
    if st.button("Last 7 Days"):
        df = df.tail(7)

with colf2:
    if st.button("Full Data"):
        df = df.copy()

# =========================
# SIDEBAR (SLICERS - KEEPING YOUR ORIGINAL)
# =========================
st.sidebar.header("🎛 Filters")

date_range = st.sidebar.date_input("Select Date Range",
    [df["Date"].min(), df["Date"].max()])

day_filter = st.sidebar.multiselect("Select Day",
    df["Day"].unique(), default=df["Day"].unique())

week_filter = st.sidebar.multiselect("Select Week",
    df["Week"].unique(), default=df["Week"].unique())

source_filter = st.sidebar.multiselect("Review Source",
    reviews[platform].dropna().unique(),
    default=reviews[platform].dropna().unique())

# =========================
# FILTER DATA
# =========================
filtered = df[
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1])) &
    (df["Day"].isin(day_filter)) &
    (df["Week"].isin(week_filter))
]

filtered_reviews = reviews[reviews[platform].isin(source_filter)]

# =========================
# KPI
# =========================
st.subheader("📊 Key Insights")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Collection", round(filtered["Daily Collection (Cr)"].sum(),2))
c2.metric("Average Sentiment", round(filtered["Sentiment Score"].mean(),2))
c3.metric("Total Reviews", len(filtered_reviews))
c4.metric("Peak Collection Day", round(filtered["Daily Collection (Cr)"].max(),2))

# =========================
# VISUALS
# =========================
tab1, tab2, tab3 = st.tabs(["Overview","Sentiment","Revenue"])

with tab1:
    st.plotly_chart(px.bar(filtered, x="Date", y="Daily Collection (Cr)",
                           color="Collection Category",
                           title="Daily Box Office Trend"),
                    use_container_width=True)

    st.plotly_chart(px.scatter(filtered,
                               x="Sentiment Score",
                               y="Daily Collection (Cr)",
                               size="Daily Collection (Cr)",
                               color="Collection Category",
                               title="Sentiment vs Revenue"),
                    use_container_width=True)

with tab2:
    st.plotly_chart(px.line(filtered, x="Date", y="Sentiment Score",
                            title="Sentiment Trend"),
                    use_container_width=True)

    st.plotly_chart(px.histogram(filtered_reviews,
                                 x=rating,
                                 title="User Rating Distribution"),
                    use_container_width=True)

with tab3:
    st.plotly_chart(px.bar(weekwise,
                           x=find_col(weekwise,"week"),
                           y=find_numeric(weekwise,["collection","cr"]),
                           title="Weekly Revenue"),
                    use_container_width=True)

    weekend = filtered[filtered["Day"].isin(["Saturday","Sunday"])]["Daily Collection (Cr)"].mean()
    weekday = filtered[~filtered["Day"].isin(["Saturday","Sunday"])]["Daily Collection (Cr)"].mean()

    st.plotly_chart(px.bar(
        pd.DataFrame({"Type":["Weekend","Weekday"],"Collection":[weekend,weekday]}),
        x="Type", y="Collection",
        title="Weekend vs Weekday Performance"
    ), use_container_width=True)

# =========================
# INSIGHTS
# =========================
st.subheader("📌 Business Insights")

corr = filtered[["Daily Collection (Cr)", "Sentiment Score"]].corr().iloc[0,1]

st.write(f"Correlation: {round(corr,3)}")

if corr > 0.6:
    st.success("Strong positive impact of reviews on revenue")
elif corr > 0.3:
    st.warning("Moderate relationship observed")
else:
    st.error("Weak relationship")
