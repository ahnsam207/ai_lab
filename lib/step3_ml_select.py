import streamlit as st
from lab_utils import go_next, go_prev, render_visualization

ALGO_BY_TYPE = {
    "분류": ["로지스틱 회귀", "랜덤포레스트", "SVM"],
    "회귀": ["선형회귀", "랜덤포레스트회귀"],
    "군집": ["KMeans"],
}

BASE_CODE = {
    "로지스틱 회귀": "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(max_iter=1000)",
    "랜덤포레스트": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)",
    "SVM": "from sklearn.svm import SVC\nmodel = SVC(probability=True)",
    "선형회귀": "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()",
    "랜덤포레스트회귀": "from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor(n_estimators=100, random_state=42)",
    "KMeans": "from sklearn.cluster import KMeans\nmodel = KMeans(n_clusters=3, random_state=42, n_init=10)",
}


def render():
    st.subheader("3단계 — 머신러닝 선택")

    df = st.session_state["df"]
    if df is None:
        st.warning("1단계에서 데이터를 먼저 준비해주세요.")
        return

    problem_type = st.radio("문제 유형 (필수)", list(ALGO_BY_TYPE.keys()))
    algorithm = st.radio("알고리즘 선택 (필수)", ALGO_BY_TYPE[problem_type])

    target_col = None
    if problem_type != "군집":
        target_col = st.radio("타겟(예측) 컬럼 선택 (필수)", df.columns.tolist())

    st.markdown("### 추천 코드 미리보기")
    st.code(BASE_CODE[algorithm], language="python")

    st.session_state["problem_type"] = problem_type
    st.session_state["algorithm"] = algorithm
    st.session_state["target_col"] = target_col

    st.markdown("---")
    render_visualization(df, key_prefix="step3_viz")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← 이전 단계"):
            go_prev()
    with c2:
        if st.button("다음 단계 →", type="primary"):
            go_next()
