import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Career Advisor", layout="centered")
st.title("💼 AI Career Advisor")
st.write("Enter your skills and discover your top job matches with recommended courses!")

jobs = pd.read_csv("jobs_dataset.csv")
jobs['Required Skills'] = jobs['Required Skills'].str.lower()

user_skills = st.text_input("Enter your skills separated by commas (e.g., Python, Excel, SQL, Analysis, APIs):")

if user_skills:
    user_skill_set = set([s.strip() for s in user_skills.lower().split(",")])


    all_skills = jobs['Required Skills'].tolist() + [user_skills]
    vectorizer = CountVectorizer().fit_transform(all_skills)
    vectors = vectorizer.toarray()
    job_vectors = vectors[:-1]
    user_vector = vectors[-1]
    similarities = cosine_similarity([user_vector], job_vectors)[0]
    jobs['Match %'] = similarities * 100


    def missing_skills(row):
        job_skill_set = set([s.strip() for s in row['Required Skills'].split(",")])
        missing = job_skill_set - user_skill_set
        return ", ".join(missing) if missing else "None"

    jobs['Missing Skills'] = jobs.apply(missing_skills, axis=1)


    course_suggestions = {
        "python": "Python for Beginners",
        "sql": "SQL Fundamentals",
        "excel": "Advanced Excel",
        "data visualization": "Data Visualization with Python",
        "machine learning": "Intro to Machine Learning",
        "statistics": "Statistics Basics",
        "html": "HTML & CSS Crash Course",
        "javascript": "JavaScript Essentials",
        "deep learning": "Deep Learning Specialization",
        "problem solving": "Problem Solving Techniques",
        "communication": "Effective Communication Skills",
        "power bi": "Power BI Essential Training",
        "neural networks": "Neural Networks for AI",
        "data analysis": "Data Analysis with Python"
    }

    def suggest_courses(missing_skills):
        if missing_skills == "None":
            return "You have all required skills!"
        courses = []
        for skill in [s.strip() for s in missing_skills.split(",")]:
            if skill in course_suggestions:
                courses.append(course_suggestions[skill])
        return ", ".join(courses) if courses else "No course suggestions"

    jobs['Suggested Courses'] = jobs['Missing Skills'].apply(suggest_courses)


    recommended_jobs = jobs.sort_values(by='Match %', ascending=False)


    st.subheader("Top Job Matches:")

    st.dataframe(recommended_jobs[['Job Title', 'Match %', 'Missing Skills', 'Suggested Courses']].head(10))
