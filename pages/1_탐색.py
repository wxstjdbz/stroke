import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------
# 기본 설정
# ---------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.title("🔎 데이터 탐색")

st.write(
    "뇌졸중 데이터의 나이, 평균 혈당, 고혈압, 심장병, "
    "BMI, 흡연 상태 등을 그래프와 표로 살펴봅니다."
)

# ---------------------------
# 데이터 불러오기
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()

# ---------------------------
# 1. 나이와 평균 혈당 분포
# ---------------------------
st.subheader("1. 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이의 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        }
    )

    fig_age.update_layout(
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )

with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당의 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        }
    )

    fig_glucose.update_layout(
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )

# ---------------------------
# 2. 뇌졸중 여부에 따른 비교
# ---------------------------
st.subheader("2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교")

plot_df = df.copy()

plot_df["stroke_label"] = plot_df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

col1, col2 = st.columns(2)

with col1:
    fig_age_box = px.box(
        plot_df,
        x="stroke_label",
        y="age",
        title="뇌졸중 여부에 따른 나이",
        labels={
            "stroke_label": "뇌졸중 여부",
            "age": "나이"
        }
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True
    )

with col2:
    fig_glucose_box = px.box(
        plot_df,
        x="stroke_label",
        y="avg_glucose_level",
        title="뇌졸중 여부에 따른 평균 혈당",
        labels={
            "stroke_label": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        }
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True
    )

# 평균값 표
age_mean = (
    plot_df
    .groupby("stroke_label")["age"]
    .mean()
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
)

glucose_mean = (
    plot_df
    .groupby("stroke_label")["avg_glucose_level"]
    .mean()
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
)

mean_table = pd.DataFrame({
    "뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"],
    "나이 평균": age_mean.values,
    "평균 혈당 평균": glucose_mean.values
})

mean_table["나이 평균"] = mean_table["나이 평균"].round(2)
mean_table["평균 혈당 평균"] = mean_table["평균 혈당 평균"].round(2)

st.write("**두 그룹의 평균값**")

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True
)

# ---------------------------
# 3. 고혈압에 따른 뇌졸중 비율
# ---------------------------
st.subheader("3. 고혈압 여부에 따른 뇌졸중 비율")

hypertension_rate = (
    df.groupby("hypertension")["stroke"]
    .mean()
    .mul(100)
    .reset_index()
)

hypertension_rate["고혈압 여부"] = hypertension_rate["hypertension"].map({
    0: "고혈압 없음",
    1: "고혈압 있음"
})

fig_hypertension = px.bar(
    hypertension_rate,
    x="고혈압 여부",
    y="stroke",
    title="고혈압 여부에 따른 뇌졸중 비율",
    labels={
        "고혈압 여부": "고혈압 여부",
        "stroke": "뇌졸중 비율(%)"
    },
    text="stroke"
)

fig_hypertension.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_hypertension.update_layout(
    yaxis_title="뇌졸중 비율(%)"
)

st.plotly_chart(
    fig_hypertension,
    use_container_width=True
)

# ---------------------------
# 4. 심장병에 따른 뇌졸중 비율
# ---------------------------
st.subheader("4. 심장병 여부에 따른 뇌졸중 비율")

heart_rate = (
    df.groupby("heart_disease")["stroke"]
    .mean()
    .mul(100)
    .reset_index()
)

heart_rate["심장병 여부"] = heart_rate["heart_disease"].map({
    0: "심장병 없음",
    1: "심장병 있음"
})

fig_heart = px.bar(
    heart_rate,
    x="심장병 여부",
    y="stroke",
    title="심장병 여부에 따른 뇌졸중 비율",
    labels={
        "심장병 여부": "심장병 여부",
        "stroke": "뇌졸중 비율(%)"
    },
    text="stroke"
)

fig_heart.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_heart.update_layout(
    yaxis_title="뇌졸중 비율(%)"
)

st.plotly_chart(
    fig_heart,
    use_container_width=True
)

# ---------------------------
# 5. BMI 빈 값인 사람들의 뇌졸중 비율
# ---------------------------
st.subheader("5. BMI가 비어 있는 사람들의 뇌졸중 비율")

bmi_missing = df[df["bmi"].isna()]

bmi_missing_stroke_rate = bmi_missing["stroke"].mean() * 100
total_stroke_rate = df["stroke"].mean() * 100

bmi_table = pd.DataFrame({
    "구분": [
        "BMI가 비어 있는 사람",
        "전체 사람"
    ],
    "사람 수": [
        len(bmi_missing),
        len(df)
    ],
    "뇌졸중 비율": [
        f"{bmi_missing_stroke_rate:.2f}%",
        f"{total_stroke_rate:.2f}%"
    ]
})

st.dataframe(
    bmi_table,
    use_container_width=True,
    hide_index=True
)

# ---------------------------
# 6. 흡연 상태별 사람 수
# ---------------------------
st.subheader("6. 흡연 상태별 사람 수")

smoking_count = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

smoking_count["흡연 상태"] = smoking_count["흡연 상태"].fillna("빈 값")

st.dataframe(
    smoking_count,
    use_container_width=True,
    hide_index=True
)
