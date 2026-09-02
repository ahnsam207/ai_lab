import streamlit as st
import pandas as pd
import numpy as np
from lab_utils import run_code, code_box, go_next, go_prev, show_dataframe_fit


def _null_handling_code(df, method):
    null_cols = df.columns[df.isnull().any()].tolist()
    if not null_cols:
        return "# 결측치가 없어 처리할 항목이 없습니다.\nresult_df = df.copy()"
    num_cols = [c for c in null_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in null_cols if c not in num_cols]

    if method == "평균/중앙값 대체":
        return (
            "result_df = df.copy()\n"
            f"num_cols = {num_cols}\n"
            "result_df[num_cols] = result_df[num_cols].fillna(result_df[num_cols].mean())"
        )
    if method == "최빈값 대체":
        return (
            "result_df = df.copy()\n"
            f"target_cols = {null_cols}\n"
            "for c in target_cols:\n"
            "    result_df[c] = result_df[c].fillna(result_df[c].mode()[0])"
        )
    if method == "행 제거":
        return "result_df = df.dropna()"
    return "result_df = df.copy()  # 처리 안 함"


def render():
    st.subheader("2단계 — 전처리")

    df = st.session_state["df"]
    if df is None:
        st.warning("1단계에서 데이터를 먼저 준비해주세요.")
        return

    st.caption(f"현재 결측치 총 {df.isnull().sum().sum()}개")

    method = st.radio(
        "결측치 처리 방식 (필수)",
        ["평균/중앙값 대체", "최빈값 대체", "행 제거", "처리 안 함"],
    )
    base_code = _null_handling_code(df, method)
    code = code_box("null_handling", base_code)
    if st.button("결측치 처리 실행", type="primary"):
        try:
            result = run_code(code, {"df": df})
            st.session_state["df"] = result["result_df"]
            st.success("결측치 처리 완료")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("실행 전")
                show_dataframe_fit(df.head())
            with c2:
                st.caption("실행 후")
                show_dataframe_fit(result["result_df"].head())
        except Exception as e:
            st.error(f"실행 오류: {e}")

    df = st.session_state["df"]  # 갱신된 df 사용

    st.markdown("### 추가 전처리 (선택)")
    do_encode = st.checkbox("범주형 인코딩")
    encode_method = None
    if do_encode:
        encode_method = st.radio("인코딩 방식", ["라벨 인코딩", "원-핫 인코딩"], key="encode_method")

    do_scale = st.checkbox("스케일링")
    scale_method = None
    if do_scale:
        scale_method = st.radio("스케일링 방식", ["StandardScaler", "MinMaxScaler"], key="scale_method")

    do_outlier = st.checkbox("이상치 제거 (IQR 기준)")
    do_dup = st.checkbox("중복 행 제거")

    if do_encode:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            st.info("범주형 컬럼이 없습니다.")
        else:
            if encode_method == "라벨 인코딩":
                code = (
                    "from sklearn.preprocessing import LabelEncoder\n"
                    "result_df = df.copy()\n"
                    f"cat_cols = {cat_cols}\n"
                    "for c in cat_cols:\n"
                    "    result_df[c] = LabelEncoder().fit_transform(result_df[c].astype(str))"
                )
            else:
                code = f"result_df = pd.get_dummies(df, columns={cat_cols})"
            code = code_box("encode", code)
            if st.button("인코딩 실행"):
                try:
                    result = run_code(code, {"df": df})
                    st.session_state["df"] = result["result_df"]
                    st.success("인코딩 완료")
                    show_dataframe_fit(result["result_df"].head())
                except Exception as e:
                    st.error(f"실행 오류: {e}")
            df = st.session_state["df"]

    if do_scale:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("숫자형 컬럼이 없습니다.")
        else:
            scaler_cls = "StandardScaler" if scale_method == "StandardScaler" else "MinMaxScaler"
            code = (
                f"from sklearn.preprocessing import {scaler_cls}\n"
                "result_df = df.copy()\n"
                f"num_cols = {num_cols}\n"
                f"result_df[num_cols] = {scaler_cls}().fit_transform(result_df[num_cols])"
            )
            code = code_box("scale", code)
            if st.button("스케일링 실행"):
                try:
                    result = run_code(code, {"df": df})
                    st.session_state["df"] = result["result_df"]
                    st.success("스케일링 완료")
                    show_dataframe_fit(result["result_df"].head())
                except Exception as e:
                    st.error(f"실행 오류: {e}")
            df = st.session_state["df"]

    if do_outlier:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        code = (
            "result_df = df.copy()\n"
            f"num_cols = {num_cols}\n"
            "for c in num_cols:\n"
            "    q1, q3 = result_df[c].quantile(0.25), result_df[c].quantile(0.75)\n"
            "    iqr = q3 - q1\n"
            "    result_df = result_df[(result_df[c] >= q1 - 1.5*iqr) & (result_df[c] <= q3 + 1.5*iqr)]"
        )
        code = code_box("outlier", code)
        if st.button("이상치 제거 실행"):
            try:
                result = run_code(code, {"df": df})
                st.session_state["df"] = result["result_df"]
                st.success(f"이상치 제거 완료 — {df.shape[0]}행 → {result['result_df'].shape[0]}행")
                show_dataframe_fit(result["result_df"].head())
            except Exception as e:
                st.error(f"실행 오류: {e}")
        df = st.session_state["df"]

    if do_dup:
        dup_count = df.duplicated().sum()
        code = code_box("dup", "result_df = df.drop_duplicates()")
        if st.button("중복 제거 실행"):
            try:
                result = run_code(code, {"df": df})
                st.session_state["df"] = result["result_df"]
                st.success(f"중복 {dup_count}건 제거 완료")
                show_dataframe_fit(result["result_df"].head())
            except Exception as e:
                st.error(f"실행 오류: {e}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← 이전 단계"):
            go_prev()
    with c2:
        if st.button("다음 단계 →", type="primary"):
            go_next()
