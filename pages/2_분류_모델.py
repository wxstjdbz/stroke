import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------
# 기본 설정
# ---------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

FEATURE_COLUMNS = list(FEATURE_NAMES.keys())

st.title("🤖 분류 모델")

st.write(
    "뇌졸중 여부를 예측하는 두 가지 분류 모델을 만들어 봅니다. "
    "stroke가 1이면 뇌졸중(양성), 0이면 뇌졸중 아님(음성)입니다."
)


# ---------------------------
# 데이터 불러오기
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()


# ---------------------------
# 입력 속성 선택
# ---------------------------
st.subheader("1. 입력 속성 고르기")

selected_features = st.multiselect(
    "모델의 입력으로 사용할 속성을 고르세요.",
    options=FEATURE_COLUMNS,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 골라 주세요.")
    st.stop()

selected_feature_names = [
    FEATURE_NAMES[feature]
    for feature in selected_features
]

st.write(
    "선택한 속성: " + ", ".join(selected_feature_names)
)


# ---------------------------
# 번호 순으로 정렬하고
# 열 명씩 묶어서 앞 세 명을 테스트용으로 고정
# ---------------------------
model_df = df.sort_values("id").reset_index(drop=True).copy()

group_position = model_df.index % 10

test_mask = group_position < 3

test_df = model_df.loc[test_mask].copy()
train_df = model_df.loc[~test_mask].copy()

# 테스트용 데이터는 1,533명
st.write(
    f"훈련용 데이터: {len(train_df):,}명 / "
    f"테스트용 데이터: {len(test_df):,}명"
)


# ---------------------------
# X, y 만들기
# ---------------------------
X_train_raw = train_df[selected_features].copy()
X_test_raw = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)


# ---------------------------
# BMI를 고른 경우에만
# 훈련용 중앙값으로 빈 값 채우기
# ---------------------------
imputer = None

if "bmi" in selected_features:
    imputer = SimpleImputer(strategy="median")

    X_train_raw[["bmi"]] = imputer.fit_transform(
        X_train_raw[["bmi"]]
    )

    X_test_raw[["bmi"]] = imputer.transform(
        X_test_raw[["bmi"]]
    )


# ---------------------------
# 크기 맞추기
# 훈련용 데이터로만 기준을 계산
# ---------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)


# ---------------------------
# 모델 만들기
# ---------------------------
logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

baseline_model = DummyClassifier(
    strategy="most_frequent"
)


# ---------------------------
# 모델 학습
# ---------------------------
logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
baseline_model.fit(X_train, y_train)


# ---------------------------
# 정확도 계산
# ---------------------------
logistic_train_accuracy = accuracy_score(
    y_train,
    logistic_model.predict(X_train)
)

logistic_test_accuracy = accuracy_score(
    y_test,
    logistic_model.predict(X_test)
)

tree_train_accuracy = accuracy_score(
    y_train,
    tree_model.predict(X_train)
)

tree_test_accuracy = accuracy_score(
    y_test,
    tree_model.predict(X_test)
)

baseline_train_accuracy = accuracy_score(
    y_train,
    baseline_model.predict(X_train)
)

baseline_test_accuracy = accuracy_score(
    y_test,
    baseline_model.predict(X_test)
)


# ---------------------------
# 정확도 카드
# ---------------------------
st.subheader("2. 모델 정확도")

card1, card2, card3 = st.columns(3)


def accuracy_card(
    container,
    title,
    test_accuracy,
    train_accuracy
):
    with container:
        st.markdown(f"### {title}")
        st.metric(
            "테스트 데이터 정확도",
            f"{test_accuracy * 100:.2f}%"
        )

        small_col1, small_col2 = st.columns(2)

        with small_col1:
            st.caption("훈련 데이터")
            st.write(f"**{train_accuracy * 100:.2f}%**")

        with small_col2:
            st.caption("테스트 데이터")
            st.write(f"**{test_accuracy * 100:.2f}%**")


accuracy_card(
    card1,
    "로지스틱 회귀(확률로 답하는 모델)",
    logistic_test_accuracy,
    logistic_train_accuracy
)

accuracy_card(
    card2,
    "의사결정트리(질문으로 답하는 모델)",
    tree_test_accuracy,
    tree_train_accuracy
)

accuracy_card(
    card3,
    "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델",
    baseline_test_accuracy,
    baseline_train_accuracy
)


# ---------------------------
# 시각화에 사용할 두 속성 선택
# ---------------------------
st.subheader("3. 두 속성으로 모델 살펴보기")

if len(selected_features) == 2:
    x_feature = selected_features[0]
    y_feature = selected_features[1]
else:
    col1, col2 = st.columns(2)

    with col1:
        x_feature = st.selectbox(
            "가로축",
            options=selected_features,
            format_func=lambda x: FEATURE_NAMES[x],
            index=0
        )

    remaining_features = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    with col2:
        y_feature = st.selectbox(
            "세로축",
            options=remaining_features,
            format_func=lambda x: FEATURE_NAMES[x]
        )


st.write(
    f"가로축: **{FEATURE_NAMES[x_feature]}** / "
    f"세로축: **{FEATURE_NAMES[y_feature]}**"
)


# ---------------------------
# 나머지 속성의 기준값
# 테스트 데이터 중앙값
# ---------------------------
plot_features = [x_feature, y_feature]

other_features = [
    feature
    for feature in selected_features
    if feature not in plot_features
]

median_values = {}

for feature in other_features:
    median_values[feature] = test_df[feature].median()

    st.write(
        f"{FEATURE_NAMES[feature]}은(는) "
        f"테스트 데이터의 중앙값 **{median_values[feature]:.4f}**에 "
        f"고정해서 계산했습니다."
    )


# ---------------------------
# 모델 입력용 행렬을 만드는 함수
# ---------------------------
def make_feature_dataframe(x_values, y_values):
    result = pd.DataFrame(
        {
            feature: 0.0
            for feature in selected_features
        }
    )

    result[x_feature] = x_values
    result[y_feature] = y_values

    for feature in other_features:
        result[feature] = median_values[feature]

    return result[selected_features]


# ---------------------------
# 테스트 데이터용 원래 값
# ---------------------------
plot_test = test_df.copy()

x_values = plot_test[x_feature].to_numpy()
y_values = plot_test[y_feature].to_numpy()


# ---------------------------
# 그림의 범위
# ---------------------------
x_min = float(test_df[x_feature].min())
x_max = float(test_df[x_feature].max())
y_min = float(test_df[y_feature].min())
y_max = float(test_df[y_feature].max())

x_margin = (x_max - x_min) * 0.08
y_margin = (y_max - y_min) * 0.08

if x_margin == 0:
    x_margin = 1

if y_margin == 0:
    y_margin = 1

plot_x_min = x_min - x_margin
plot_x_max = x_max + x_margin
plot_y_min = y_min - y_margin
plot_y_max = y_max + y_margin


# ---------------------------
# 격자 생성
# ---------------------------
grid_size = 120

grid_x = pd.Series(
    pd.np.linspace(
        plot_x_min,
        plot_x_max,
        grid_size
    )
)

grid_y = pd.Series(
    pd.np.linspace(
        plot_y_min,
        plot_y_max,
        grid_size
    )
)
