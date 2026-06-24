import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

file_path = r"F:\Guvi\youtube_ad_revenue_dataset.csv"
df = pd.read_csv(file_path)
df.head()


model = pickle.load(
    open('youtube_revenue_model.pkl', 'rb')
)

page = st.sidebar.radio(
    "Content Monetization Modeler",
    ["Model Performance","Revenue prediction"]
)   

if page == "Revenue prediction":

    st.title("Content Monetization Modeler")
    st.header("Youtube Ad revenue Predictor")   


    views = st.number_input("Views", min_value=0)

    likes = st.number_input("Likes", min_value=0)

    comments = st.number_input("Comments", min_value=0)

    watch_time_minutes = st.number_input("Watch Time Minutes", min_value=0.0)

    video_length_minutes = st.number_input("Video Length Minutes", min_value=0.0)

    subscribers = st.number_input("Subscribers", min_value=0)

    category = st.selectbox(
        "Category",
        ["Education", "Gaming", "Music", "Tech","Entertainment","Lifestyle"]
    )

    device = st.selectbox(
        "device",
        ["Desktop", "Mobile", "Tablet", "TV"]
    )

    country = st.selectbox(
        "country",
        ["AU", "CA", "DE", "IN","UK","US"]
    )

    model_columns = pickle.load(open('model_columns.pkl', 'rb'))

    input_df = pd.DataFrame([[0] * len(model_columns)],columns = model_columns)

    input_df['views'] = views
    input_df['likes'] = likes
    input_df['comments'] = comments
    input_df['watch_time_minutes'] = watch_time_minutes
    input_df['video_length_minutes'] = video_length_minutes
    input_df['subscribers'] = subscribers

    if views > 0:
        input_df['engagement_rate'] = (likes + comments) / views
    else:
        input_df['engagement_rate'] = 0

    col = f"category_{category}"

    if col in input_df.columns:
        input_df[col] = 1   

    col = f"device_{device}"

    if col in input_df.columns:
        input_df[col] = 1  

    col = f"country_{country}"

    if col in input_df.columns:
        input_df[col] = 1      


    if st.button("Predict Revenue"):

        prediction = model.predict(input_df)

        st.success(
            f"Predicted Revenue: ${prediction[0]:.2f}"
        )


elif page == "Model Performance":

    st.title("Content Monetization Modeler")
    st.header("Model Performance")

    st.success(
    "Best Model Selected: Linear Regression "
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", "0.9526")
    col2.metric("MAE", "3.10")
    col3.metric("RMSE", "13.48")

    st.subheader("Insights")

    st.markdown("""
    - Watch Time showed the strongest positive relationship with revenue.
    - Likes and Comments positively influenced revenue prediction.
    - Technology category videos tended to generate higher revenue.
    - Mobile users showed higher revenue contribution compared to TV users.
    - India showed the strongest positive contribution among countries.
    """)

    st.subheader("Watch Time vs Revenue")

    sample_df = df.sample(3000, random_state=42)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        sample_df['watch_time_minutes'],
        sample_df['ad_revenue_usd'],
        alpha=0.4
    )

    ax.set_xlabel("Watch Time Minutes")
    ax.set_ylabel("Revenue (USD)")
    ax.set_title("Watch Time vs Revenue")

    st.pyplot(fig)
        

    results = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Gradient Boosting',
        'Random Forest',
        'KNN',
        'Decision Tree'
    ],

    'R2 Score': [
        0.9526,
        0.9523,
        0.9499,
        0.9291,
        0.9037
    ],

    'MAE': [
        3.10,
        3.62,
        3.59,
        8.28,
        5.26
    ],

    'RMSE': [
        13.48,
        13.52,
        13.85,
        16.48,
        19.21
    ]
    })

    st.dataframe(results)

    comparison = pd.read_csv(
    'actual_vs_predicted.csv'
    )

    st.subheader(
        "Sample Actual vs Predicted Revenue"
    )

    st.dataframe(comparison.head(20))

    
