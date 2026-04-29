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
# COLUMN DETECTION
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

if platform is None:
    reviews["review_source"] = "All"
    platform = "review_source"

# =========================
# CLEAN DATA
# =========================
daywise[date_day] = pd.to_datetime(daywise[date_day])
sentiment[date_sent] = pd.to_datetime(sentiment[date_sent])
reviews[date_review] = pd.to_datetime(reviews[date_review])

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
# 🎛 BUTTON SLICERS
# =========================
st.subheader("⚡ Quick Filters")

col1, col2, col3, col4 = st.columns(4)

selected_day = "All"
selected_week = "All"
selected_source = "All"

with col1:
    selected_day = st.selectbox("Day", ["All"] + list(df["Day"].unique()))

with col2:
    selected_week = st.selectbox("Week", ["All"] + list(df["Week"].unique()))

with col3:
    selected_source = st.selectbox("Source", ["All"] + list(reviews[platform].dropna().unique()))

with col4:
    if st.button("Last 7 Days"):
        df = df.tail(7)

# APPLY FILTERS
filtered = df.copy()

if selected_day != "All":
    filtered = filtered[filtered["Day"] == selected_day]

if selected_week != "All":
    filtered = filtered[filtered["Week"] == selected_week]

filtered_reviews = reviews.copy()
if selected_source != "All":
    filtered_reviews = reviews[reviews[platform] == selected_source]

# =========================
# KPI
# =========================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Collection", round(filtered["Daily Collection (Cr)"].sum(),2))
c2.metric("Avg Sentiment", round(filtered["Sentiment Score"].mean(),2))
c3.metric("Reviews", len(filtered_reviews))
c4.metric("Peak Day", round(filtered["Daily Collection (Cr)"].max(),2))

# =========================
# VISUALS
# =========================
tab1, tab2, tab3 = st.tabs(["Overview","Sentiment","Revenue"])

with tab1:
    st.plotly_chart(px.bar(filtered, x="Date", y="Daily Collection (Cr)",
                           color="Collection Category",
                           color_discrete_sequence=px.colors.qualitative.Bold,
                           title="Daily Collection Trend"),
                    use_container_width=True)

    st.plotly_chart(px.scatter(filtered,
                               x="Sentiment Score",
                               y="Daily Collection (Cr)",
                               color="Collection Category",
                               size="Daily Collection (Cr)",
                               color_discrete_sequence=px.colors.qualitative.Vivid,
                               title="Sentiment vs Revenue"),
                    use_container_width=True)

    # NEW VISUAL
    st.plotly_chart(px.area(filtered, x="Date", y="Daily Collection (Cr)",
                            title="Revenue Growth Over Time"),
                    use_container_width=True)

with tab2:
    st.plotly_chart(px.line(filtered, x="Date", y="Sentiment Score",
                            color_discrete_sequence=["cyan"],
                            title="Sentiment Trend"),
                    use_container_width=True)

    st.plotly_chart(px.histogram(filtered_reviews,
                                 x=rating,
                                 color=platform,
                                 title="Ratings by Source"),
                    use_container_width=True)

    # NEW VISUAL
    st.plotly_chart(px.box(filtered_reviews,
                           x=platform,
                           y=rating,
                           title="Rating Spread by Platform"),
                    use_container_width=True)

with tab3:
    st.plotly_chart(px.bar(weekwise,
                           x=find_col(weekwise,"week"),
                           y=find_numeric(weekwise,["collection","cr"]),
                           color_discrete_sequence=["orange"],
                           title="Weekly Revenue"),
                    use_container_width=True)

    # NEW VISUAL
    st.plotly_chart(px.pie(filtered,
                           names="Collection Category",
                           title="Revenue Distribution by Category"),
                    use_container_width=True)

    weekend = filtered[filtered["Day"].isin(["Saturday","Sunday"])]["Daily Collection (Cr)"].mean()
    weekday = filtered[~filtered["Day"].isin(["Saturday","Sunday"])]["Daily Collection (Cr)"].mean()

    st.plotly_chart(px.bar(
        pd.DataFrame({"Type":["Weekend","Weekday"],"Collection":[weekend,weekday]}),
        x="Type", y="Collection",
        color="Type",
        title="Weekend vs Weekday"
    ), use_container_width=True)
