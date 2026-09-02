import streamlit as st
import pandas as pd
from sklearn import datasets as sk_datasets
from lab_utils import run_code, code_box, go_next, show_named_series, show_dataframe_fit

SAMPLE_SETS = {
    "iris (분류용)": lambda: sk_datasets.load_iris(as_frame=True).frame,
    "wine (분류용)": lambda: sk_datasets.load_wine(as_frame=True).frame,
    "california_housing (회귀용)": lambda: sk_datasets.fetch_california_housing(as_frame=True).frame,
}


def render():
    st.subheader("1단계 — 데이터 입력")

    source = st.radio("데이터 소스 선택 (필수)", ["파일 업로드", "샘플 데이터 사용"])

    df = None
    if source == "파일 업로드":
        uploaded = st.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx"])
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
            except Exception as e:
                st.error(f"파일을 읽는 중 오류: {e}")
    else:
        sample_name = st.selectbox("샘플 데이터셋 선택", list(SAMPLE_SETS.keys()))
        if st.button("샘플 데이터 불러오기"):
            try:
                df = SAMPLE_SETS[sample_name]()
            except Exception as e:
                st.error(f"샘플 데이터 로드 오류: {e}")

    if df is not None:
        st.session_state["df"] = df

    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.success(f"데이터 준비 완료 — {df.shape[0]}행 × {df.shape[1]}열")
        show_dataframe_fit(df.head())

        st.markdown("### 데이터 확인 옵션 (선택)")
        opt_describe = st.checkbox("기술통계 보기 (df.describe())")
        opt_null = st.checkbox("결측치 요약 보기 (df.isnull().sum())")
        opt_dtype = st.checkbox("데이터 타입 보기 (df.dtypes)")

        if opt_describe:
            code = code_box("describe", "df.describe()")
            if st.button("기술통계 실행", key="run_describe"):
                try:
                    result = run_code(f"result = {code}", {"df": df})
                    show_dataframe_fit(result["result"], max_width=900)
                except Exception as e:
                    st.error(f"실행 오류: {e}")

        if opt_null:
            code = code_box("nullsum", "df.isnull().sum()")
            if st.button("결측치 요약 실행", key="run_null"):
                try:
                    result = run_code(f"result = {code}", {"df": df})
                    show_named_series(result["result"], index_label="컬럼", value_label="결측치 수")
                except Exception as e:
                    st.error(f"실행 오류: {e}")

        if opt_dtype:
            code = code_box("dtypes", "df.dtypes")
            if st.button("데이터 타입 보기 실행", key="run_dtype"):
                try:
                    result = run_code(f"result = {code}", {"df": df})
                    show_named_series(result["result"].astype(str), index_label="컬럼", value_label="데이터 타입")
                except Exception as e:
                    st.error(f"실행 오류: {e}")

        st.markdown("---")
        if st.button("다음 단계 →", type="primary"):
            go_next()
    else:
        st.info("데이터를 업로드하거나 샘플 데이터를 불러오면 다음 단계로 진행할 수 있습니다.")
