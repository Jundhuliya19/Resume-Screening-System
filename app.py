import streamlit as st
import pandas as pd
import plotly.express as px
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Screening System")
st.markdown("### Intelligent Resume Ranking using Machine Learning")
st.write("---")

st.sidebar.title("⚙ Settings")

job_description = st.sidebar.text_area(
    "Job Description",
    """Looking for a Python Developer with machine learning,
data analysis, Pandas, NumPy, SQL,
Scikit-learn, communication skills."""
)

uploaded_file = st.file_uploader(
    "📂 Upload Resume CSV",
    type=["csv"]
)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+", " ", text)
    text = re.sub(r"\\S+@\\S+", " ", text)
    text = re.sub(r"\\d+", " ", text)
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    text = " ".join(text.split())
    return text


def find_resume_column(df):

    possible = [
        "Resume",
        "resume",
        "Resume_str",
        "resume_text",
        "Text",
        "text"
    ]

    for col in possible:
        if col in df.columns:
            return col

    return df.select_dtypes(include="object").columns[0]

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    resume_column = find_resume_column(df)

    df["Clean_Resume"] = df[resume_column].astype(str).apply(clean_text)

    clean_job = clean_text(job_description)

    if st.button("🚀 Analyze Resumes", use_container_width=True):

        with st.spinner("Analyzing resumes using AI..."):

            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=3000
            )

            vectors = vectorizer.fit_transform(
                [clean_job] + df["Clean_Resume"].tolist()
            )

            similarity = cosine_similarity(
                vectors[0:1],
                vectors[1:]
            ).flatten()

            df["Score (%)"] = (similarity * 100).round(2)

            ranked = df.sort_values(
                by="Score (%)",
                ascending=False
            ).reset_index(drop=True)

            ranked.insert(0, "Rank", range(1, len(ranked)+1))

            best = ranked.iloc[0]

            st.success("✅ AI Analysis Completed!")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "📄 Total Resumes",
                len(ranked)
            )

            col2.metric(
                "🏆 Highest Score",
                f"{ranked['Score (%)'].max():.2f}%"
            )

            col3.metric(
                "📊 Average Score",
                f"{ranked['Score (%)'].mean():.2f}%"
            )

            st.write("---")

            st.subheader("🥇 Best Candidate")

            c1, c2 = st.columns(2)

            if "ID" in ranked.columns:
                c1.info(f"**ID:** {best['ID']}")

            if "Category" in ranked.columns:
                c2.info(f"**Category:** {best['Category']}")

            st.success(
                f"Similarity Score : **{best['Score (%)']}%**"
            )

            st.write("---")

            st.subheader("📊 Top 10 Resume Scores")

            top10 = ranked.head(10)

            fig = px.bar(
                top10,
                x="Rank",
                y="Score (%)",
                color="Score (%)",
                text="Score (%)",
                title="Top 10 Resume Similarity Scores"
            )

            fig.update_layout(
                xaxis_title="Resume Rank",
                yaxis_title="Similarity Score (%)",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            st.write("---")

            st.subheader("🥧 Resume Category Distribution")

            if "Category" in ranked.columns:

                pie = px.pie(
                    ranked,
                    names="Category",
                    title="Resume Categories"
                )

                st.plotly_chart(
                    pie,
                    use_container_width=True
                )

            st.write("---")

            st.subheader("📋 Top 10 Ranked Candidates")

            st.dataframe(
                top10,
                use_container_width=True
            )

            csv = ranked.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Ranked Resume CSV",
                csv,
                "Ranked_Resumes.csv",
                "text/csv",
                use_container_width=True
            )

            st.balloons()