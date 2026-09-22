import math
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

st.write(
    "선택한 속성: "
    + ", ".join(FEATURE_NAMES[feature] for feature in selected_features)
)


# ---------------------------
# 번호 순으로 정렬
# 10명씩 묶어서 앞 3명은 테스트용
# ---------------------------
model_df = df.sort_values("id").reset_index(drop=True).copy()

group_position = model_df.index % 10
test_mask = group_position < 3

test_df = model_df.loc[test_mask].copy()
train_df = model_df.loc[~test_mask].copy()

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
# BMI를 선택한 경우에만
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
# 훈련용 데이터로만 기준 계산
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
# 두 축 선택
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
# 두 축이 아닌 속성은
# 테스트 데이터 중앙값에 고정
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
        f"테스트 데이터의 중앙값 "
        f"**{median_values[feature]:.4f}**에 고정해서 계산했습니다."
    )


# ---------------------------
# 그림 입력 데이터 만들기
# ---------------------------
def make_feature_dataframe(x_values, y_values):
    result = pd.DataFrame(index=range(len(x_values)))

    for feature in selected_features:
        result[feature] = 0.0

    result[x_feature] = x_values
    result[y_feature] = y_values

    for feature in other_features:
        result[feature] = median_values[feature]

    return result[selected_features]


# ---------------------------
# 테스트 데이터
# ---------------------------
x_values = test_df[x_feature].to_numpy()
y_values = test_df[y_feature].to_numpy()


# ---------------------------
# 그림 범위
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
# 격자 만들기
# numpy 없이 Python으로 생성
# ---------------------------
grid_size = 100

grid_x = [
    plot_x_min
    + (plot_x_max - plot_x_min) * i / (grid_size - 1)
    for i in range(grid_size)
]

grid_y = [
    plot_y_min
    + (plot_y_max - plot_y_min) * i / (grid_size - 1)
    for i in range(grid_size)
]

grid_points = []

for y_value in grid_y:
    for x_value in grid_x:
        grid_points.append((x_value, y_value))


grid_x_values = [point[0] for point in grid_points]
grid_y_values = [point[1] for point in grid_points]

grid_df = make_feature_dataframe(
    grid_x_values,
    grid_y_values
)

grid_scaled = scaler.transform(grid_df)


# ---------------------------
# 로지스틱 회귀 확률
# ---------------------------
logistic_grid_probability = logistic_model.predict_proba(
    grid_scaled
)[:, 1]

logistic_probability_grid = [
    logistic_grid_probability[
        row * grid_size:(row + 1) * grid_size
    ]
    for row in range(grid_size)
]


# ---------------------------
# 의사결정트리 영역
# ---------------------------
tree_grid_prediction = tree_model.predict(
    grid_scaled
)

tree_prediction_grid = [
    tree_grid_prediction[
        row * grid_size:(row + 1) * grid_size
    ]
    for row in range(grid_size)
]


# ---------------------------
# 산점도 + 로지스틱 경계선
# + 의사결정트리 영역
# ---------------------------
st.subheader("4. 테스트 데이터와 모델의 판단 경계")

fig = go.Figure()


# 의사결정트리 영역
fig.add_trace(
    go.Contour(
        x=grid_x,
        y=grid_y,
        z=tree_prediction_grid,
        contours=dict(
            start=0,
            end=1,
            size=1,
            coloring="fill",
            showlines=False
        ),
        colorscale=[
            [0, "rgba(150, 150, 150, 0.10)"],
            [1, "rgba(255, 150, 150, 0.15)"]
        ],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 영역"
    )
)


# 로지스틱 회귀의 0.5 경계
fig.add_trace(
    go.Contour(
        x=grid_x,
        y=grid_y,
        z=logistic_probability_grid,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="lines",
            showlabels=False
        ),
        line=dict(
            width=3
        ),
        showscale=False,
        hoverinfo="skip",
        name="로지스틱 회귀 0.5 경계"
    )
)


# 실제 뇌졸중 아님
negative_mask = test_df["stroke"].to_numpy() == 0

fig.add_trace(
    go.Scatter(
        x=x_values[negative_mask],
        y=y_values[negative_mask],
        mode="markers",
        name="실제 뇌졸중 아님",
        marker=dict(
            size=7,
            opacity=0.65
        ),
        customdata=test_df.loc[
            negative_mask,
            ["id"]
        ],
        hovertemplate=(
            "ID: %{customdata[0]}<br>"
            + FEATURE_NAMES[x_feature]
            + ": %{x:.2f}<br>"
            + FEATURE_NAMES[y_feature]
            + ": %{y:.2f}<extra></extra>"
        )
    )
)


# 실제 뇌졸중
positive_mask = test_df["stroke"].to_numpy() == 1

fig.add_trace(
    go.Scatter(
        x=x_values[positive_mask],
        y=y_values[positive_mask],
        mode="markers",
        name="실제 뇌졸중",
        marker=dict(
            size=8,
            opacity=0.75
        ),
        customdata=test_df.loc[
            positive_mask,
            ["id"]
        ],
        hovertemplate=(
            "ID: %{customdata[0]}<br>"
            + FEATURE_NAMES[x_feature]
            + ": %{x:.2f}<br>"
            + FEATURE_NAMES[y_feature]
            + ": %{y:.2f}<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    height=650,
    legend_title="실제 뇌졸중 여부"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------
# 로지스틱 경계선이 그림 밖인지 확인
# ---------------------------
corner_points = []

for x_value in [
    plot_x_min,
    plot_x_max
]:
    for y_value in [
        plot_y_min,
        plot_y_max
    ]:
        corner_df = make_feature_dataframe(
            [x_value],
            [y_value]
        )

        corner_scaled = scaler.transform(corner_df)

        probability = logistic_model.predict_proba(
            corner_scaled
        )[0, 1]

        corner_points.append(probability)

if min(corner_points) > 0.5 or max(corner_points) < 0.5:
    st.info(
        "로지스틱 회귀의 0.5 경계선은 현재 그림의 범위 밖에 있습니다."
    )
else:
    st.write(
        "로지스틱 회귀의 0.5 경계선이 현재 그림의 범위 안에 있습니다."
    )


# ---------------------------
# 의사결정트리 질문 그림
# ---------------------------
st.subheader("5. 의사결정트리가 던진 질문")

tree = tree_model.tree_

feature_indices = tree.feature
thresholds = tree.threshold

# 훈련용 데이터에서 각 노드에 도착한 사람 수와
# 실제 뇌졸중인 사람 수 계산
train_node_ids = tree_model.apply(X_train)

node_sample_count = {}
node_positive_count = {}

for node_id, target in zip(train_node_ids, y_train):
    node_sample_count[node_id] = (
        node_sample_count.get(node_id, 0) + 1
    )

    node_positive_count[node_id] = (
        node_positive_count.get(node_id, 0) + int(target)
    )


def node_statistics(node_id):
    sample_count = node_sample_count.get(node_id, 0)
    positive_count = node_positive_count.get(node_id, 0)

    if sample_count > 0:
        positive_rate = positive_count / sample_count * 100
    else:
        positive_rate = 0

    return (
        sample_count,
        positive_count,
        positive_rate
    )


def original_threshold(feature_index, threshold):
    feature_name = selected_features[feature_index]

    scale_mean = scaler.mean_[feature_index]
    scale_std = scaler.scale_[feature_index]

    return threshold * scale_std + scale_mean, feature_name


# ---------------------------
# DOT 문자열 만들기
# ---------------------------
dot_lines = [
    "digraph Tree {",
    'graph [rankdir=TB, bgcolor="transparent"];',
    'node [shape=box, style="rounded,filled", fontname="Malgun Gothic"];',
    'edge [fontname="Malgun Gothic"];'
]

leaf_nodes = []
asked_features = set()


def build_dot(node_id):
    sample_count, positive_count, positive_rate = node_statistics(
        node_id
    )

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    is_leaf = left_child == right_child

    if is_leaf:
        predicted_class = int(tree_model.tree_.value[node_id][0].argmax())

        if predicted_class == 1:
            answer = "답: 뇌졸중"
            fill_color = "#F8D7DA"
        else:
            answer = "답: 뇌졸중 아님"
            fill_color = "#D9EDF7"

        label = (
            f"훈련용: {sample_count}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {positive_rate:.1f}%\\n"
            f"{answer}"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", fillcolor="{fill_color}"];'
        )

        leaf_nodes.append(
            {
                "node": node_id,
                "prediction": predicted_class
            }
        )

        return

    feature_index = feature_indices[node_id]

    threshold, feature_name = original_threshold(
        feature_index,
        thresholds[node_id]
    )

    asked_features.add(feature_name)

    label = (
        f"{FEATURE_NAMES[feature_name]} ≤ {threshold:.2f}?\\n"
        f"훈련용: {sample_count}명\\n"
        f"실제 뇌졸중: {positive_count}명\\n"
        f"뇌졸중 비율: {positive_rate:.1f}%"
    )

    dot_lines.append(
        f'{node_id} [label="{label}", fillcolor="white"];'
    )

    build_dot(left_child)
    build_dot(right_child)

    dot_lines.append(
        f'{node_id} -> {left_child} [label="예"];'
    )

    dot_lines.append(
        f'{node_id} -> {right_child} [label="아니요"];'
    )


build_dot(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# ---------------------------
# 트리 요약
# ---------------------------
leaf_count = len(leaf_nodes)
negative_leaf_count = sum(
    leaf["prediction"] == 0
    for leaf in leaf_nodes
)

asked_feature_names = [
    FEATURE_NAMES[feature]
    for feature in selected_features
    if feature in asked_features
]

st.write(
    f"답을 내는 마디는 모두 **{leaf_count}칸**이고, "
    f"그중 **{negative_leaf_count}칸**이 "
    f"'뇌졸중 아님'이라고 답합니다."
)

if asked_feature_names:
    st.write(
        "이 나무가 실제로 물은 속성: "
        + ", ".join(asked_feature_names)
    )
else:
    st.write(
        "이 나무가 실제로 물은 속성: 없음"
    )
