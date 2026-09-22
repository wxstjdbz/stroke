import pandas as pd
import streamlit as st

# ---------------------------
# 기본 설정
# ---------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# 제목
# ---------------------------
st.title("🧠 뇌졸중 예측 실습실")

st.write(
    "이 실습실에서는 뇌졸중 예측에 사용되는 데이터를 살펴봅니다. "
    "데이터의 크기와 열의 구성, 빈 값, 실제 데이터의 모습을 확인해 보세요."
)

# ---------------------------
# 데이터 불러오기
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()

# ---------------------------
# 데이터 기본 정보
# ---------------------------
total_people = len(df)
column_count = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = stroke_count / total_people * 100

# ---------------------------
# 숫자 카드 4개
# ---------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="전체 사람 수",
        value=f"{total_people:,}명"
    )

with col2:
    st.metric(
        label="열 개수",
        value=f"{column_count}개"
    )

with col3:
    st.metric(
        label="stroke가 1인 사람 수",
        value=f"{stroke_count:,}명"
    )

with col4:
    st.metric(
        label="stroke가 1인 비율",
        value=f"{stroke_ratio:.2f}%"
    )

st.divider()

# ---------------------------
# 열 정보
# ---------------------------
st.subheader("📋 열 정보")

# 교재를 보고 '우리말 뜻' 부분을 직접 입력하세요.
korean_meanings = {
    "id": "",
    "gender": "",
    "age": "",
    "hypertension": "",
    "heart_disease": "",
    "ever_married": "",
    "work_type": "",
    "Residence_type": "",
    "avg_glucose_level": "",
    "bmi": "",
    "smoking_status": "",
    "stroke": ""
}

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [
        korean_meanings[column]
        for column in df.columns
    ],
    "값의 종류": [
        df[column].nunique(dropna=True)
        for column in df.columns
    ],
    "빈 값 개수": [
        df[column].isna().sum()
        for column in df.columns
    ]
})

st.dataframe(
    column_info,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "※ '우리말 뜻'은 교재를 참고하여 위 korean_meanings 부분에 직접 입력하세요."
)

# ---------------------------
# 데이터 처음 다섯 줄
# ---------------------------
st.subheader("🔎 데이터 처음 다섯 줄")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)

# ---------------------------
# 데이터 출처
# ---------------------------
st.divider()

st.subheader("📚 데이터 출처")

st.text_area(
    "교재에 있는 데이터 출처를 입력하세요.",
    value="",
    placeholder="여기에 교재의 데이터 출처를 입력하세요.",
    height=100
)
