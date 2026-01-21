import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI Career Guidance System",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Career Guidance System")
st.write("Career suggestion based on marks, skills and personality")

# =============================
# MACHINE LEARNING MODEL
# =============================
data = {
    'Maths':[85,70,90,60,40,88,65,75,92,55],
    'Physics':[85,60,88,65,45,80,60,70,95,55],
    'Chemistry':[80,70,85,60,50,75,65,68,92,55],
    'Computer':[90,50,95,40,30,85,60,65,98,50],
    'Biology':[70,65,60,80,85,75,70,72,65,80],
    'Accounts':[0,0,0,0,0,0,80,85,75,65],
    'Economics':[0,0,0,0,0,0,70,80,75,60],
    'Business':[0,0,0,0,0,0,75,85,80,60],
    'English':[75,80,90,60,50,78,70,72,85,60],
    'Coding':[1,0,1,0,0,1,0,0,1,0],
    'Communication':[1,1,0,1,1,0,1,1,0,1],
    'Creativity':[0,1,0,1,1,0,1,1,0,1],
    'Art':[0,1,0,1,1,0,1,1,0,1],
    'Sports':[0,0,0,0,1,0,1,1,0,1],
    'Leadership':[0,1,0,1,0,0,1,1,0,1],
    'Teamwork':[1,1,0,1,1,0,1,1,0,1],
    'ProblemSolving':[1,0,1,0,0,1,1,0,1,0],
    'Career':[
        'Software Engineer','Teacher','Data Scientist','Designer','Clerk',
        'Software Engineer','Doctor','Accountant','Entrepreneur','Sports Coach'
    ]
}

df = pd.DataFrame(data)

FEATURES = [
    'Maths','Physics','Chemistry','Computer','Biology',
    'Accounts','Economics','Business','English',
    'Coding','Communication','Creativity','Art','Sports',
    'Leadership','Teamwork','ProblemSolving'
]

X = df[FEATURES]
y = df['Career']

model = DecisionTreeClassifier()
model.fit(X, y)

# =============================
# USER INPUT
# =============================
st.header("1️⃣ Select Stream")
stream = st.radio("Choose your stream", ["Computer", "Biology", "Commerce"])

st.header("2️⃣ Enter Marks")
subjects = {
    "Computer": ['Maths','Physics','Chemistry','Computer','English'],
    "Biology": ['Maths','Physics','Chemistry','Biology','English'],
    "Commerce": ['Maths','Accounts','Economics','Business','English']
}[stream]

marks = {}
for s in subjects:
    marks[s] = st.number_input(
        f"{s} Marks",
        min_value=0,
        max_value=100,
        step=1
    )

st.header("3️⃣ Skills & Interests")
skills = {
    'Coding': st.checkbox("Coding"),
    'Communication': st.checkbox("Communication"),
    'Creativity': st.checkbox("Creativity"),
    'Art': st.checkbox("Art / Design"),
    'Sports': st.checkbox("Sports")
}

st.header("4️⃣ Personality Traits")
personality = {
    'Leadership': st.checkbox("Leadership"),
    'Teamwork': st.checkbox("Teamwork"),
    'ProblemSolving': st.checkbox("Problem Solving")
}

# =============================
# PREDICTION
# =============================
st.header("5️⃣ Career Prediction")

if st.button("🔮 Predict Career"):
    student = {f:0 for f in FEATURES}

    for k,v in marks.items():
        student[k] = v

    for k,v in skills.items():
        student[k] = int(v)

    for k,v in personality.items():
        student[k] = int(v)

    input_data = [student[f] for f in FEATURES]
    result = model.predict([input_data])[0]

    st.success("🎯 Suggested Career")
    st.markdown(f"## 🧠 {result}")

st.markdown("---")
st.caption("AI Career Guidance System | Streamlit | Free Hosting")
