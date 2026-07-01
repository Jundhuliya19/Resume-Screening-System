import re
import string
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# LOAD DATASET
# ==========================

data = pd.read_csv("Resume.csv")

print("=" * 60)
print("INTELLIGENT RESUME SCREENING SYSTEM")
print("=" * 60)

print("\nDataset Loaded Successfully!")
print("Total Resumes :", len(data))

# ==========================
# CLEAN TEXT FUNCTION
# ==========================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = " ".join(text.split())

    return text


# ==========================
# CLEAN DATA
# ==========================

resume_column = "Resume_str"

data["Clean_Resume"] = data[resume_column].apply(clean_text)

# ==========================
# JOB DESCRIPTION
# ==========================

job_description = """
Python Developer

Machine Learning

Artificial Intelligence

Data Science

Python

Pandas

NumPy

Scikit-learn

SQL

Data Analysis

Communication Skills

Problem Solving

Git

Deep Learning

"""

job_description = clean_text(job_description)

# ==========================
# TF-IDF VECTORIZATION
# ==========================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

vectors = vectorizer.fit_transform(
    [job_description] + data["Clean_Resume"].tolist()
)

# ==========================
# COSINE SIMILARITY
# ==========================

scores = cosine_similarity(
    vectors[0:1],
    vectors[1:]
).flatten()

data["Similarity Score"] = scores

data["Score (%)"] = (scores * 100).round(2)

# ==========================
# RANK RESUMES
# ==========================

ranked = data.sort_values(
    by="Similarity Score",
    ascending=False
).reset_index(drop=True)

ranked.insert(
    0,
    "Rank",
    range(1, len(ranked) + 1)
)

# ==========================
# SAVE CSV
# ==========================

ranked.to_csv(
    "Ranked_Resumes.csv",
    index=False
)

# ==========================
# DISPLAY TOP 10
# ==========================

print("\n")
print("=" * 60)
print("TOP 10 MATCHING RESUMES")
print("=" * 60)

top10 = ranked.head(10)

for _, row in top10.iterrows():

    print(
        f"Rank {row['Rank']}"
    )

    print(
        f"Category : {row['Category']}"
    )

    print(
        f"Similarity : {row['Score (%)']}%"
    )

    print("-" * 40)

# ==========================
# BEST CANDIDATE
# ==========================

best = ranked.iloc[0]

print("\n")
print("=" * 60)
print("AI HIRING RECOMMENDATION")
print("=" * 60)

print("Candidate ID :", best["ID"])

print("Category :", best["Category"])

print("Matching Score :", best["Score (%)"], "%")

if best["Score (%)"] >= 80:

    print("Recommendation : STRONGLY RECOMMENDED")

elif best["Score (%)"] >= 60:

    print("Recommendation : RECOMMENDED")

else:

    print("Recommendation : NEEDS FURTHER REVIEW")

# ==========================
# COMMON SKILLS
# ==========================

words = []

for resume in top10["Clean_Resume"]:

    words.extend(resume.split())

common = Counter(words).most_common(20)

print("\n")
print("=" * 60)
print("MOST COMMON WORDS")
print("=" * 60)

for word, count in common:

    print(f"{word:<20} {count}")

# ==========================
# BAR GRAPH
# ==========================

plt.figure(figsize=(10,5))

plt.bar(
    top10["Rank"].astype(str),
    top10["Score (%)"]
)

plt.title("Top Resume Similarity Scores")

plt.xlabel("Resume Rank")

plt.ylabel("Similarity Score (%)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Top_Resume_Scores.png",
    dpi=300
)

plt.show()

# ==========================
# PIE CHART
# ==========================

recommended = len(
    ranked[ranked["Score (%)"] >= 60]
)

others = len(ranked) - recommended

plt.figure(figsize=(6,6))

plt.pie(
    [recommended, others],
    labels=["Recommended", "Others"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Resume Screening Result")

plt.savefig(
    "Selection_Pie_Chart.png",
    dpi=300
)

plt.show()

# ==========================
# REPORT
# ==========================

with open(
    "Resume_Report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("INTELLIGENT RESUME SCREENING SYSTEM\n\n")

    file.write("TOP 10 RESUMES\n\n")

    for _, row in top10.iterrows():

        file.write(
            f"Rank {row['Rank']} | "
            f"{row['Category']} | "
            f"{row['Score (%)']}%\n"
        )

    file.write("\n")

    file.write("BEST CANDIDATE\n")

    file.write(
        f"ID : {best['ID']}\n"
    )

    file.write(
        f"Category : {best['Category']}\n"
    )

    file.write(
        f"Score : {best['Score (%)']}%\n"
    )

print("\n")
print("=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files")

print("1. Ranked_Resumes.csv")

print("2. Resume_Report.txt")

print("3. Top_Resume_Scores.png")

print("4. Selection_Pie_Chart.png")