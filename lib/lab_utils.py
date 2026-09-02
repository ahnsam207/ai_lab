"""
AI 학습 실습 사이트 - 공통 유틸리티
6단계 파이프라인의 session_state 관리와 코드 실행을 담당한다.
"""

import streamlit as st
import pandas as pd
import numpy as np

STEP_NAMES = [
    "1. 데이터 입력",
    "2. 전처리",
    "3. 머신러닝 선택",
    "4. 머신러닝 단계적 처리",
    "5. 딥러닝 선택",
    "6. 딥러닝 단계적 처리",
]


def init_session():
    defaults = {
        "step": 1,               # 현재 진행 단계 (1~6)
        "df": None,               # 원본/전처리된 데이터프레임
        "problem_type": None,     # 분류/회귀/군집
        "algorithm": None,
        "target_col": None,
        "model": None,
        "X_train": None, "X_test": None, "y_train": None, "y_test": None,
        "ml_metrics": None,
        "dl_arch": None,
        "dl_layers": None,
        "dl_model": None,
        "dl_history": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def go_next():
    st.session_state["step"] += 1
    st.rerun()


def go_prev():
    if st.session_state["step"] > 1:
        st.session_state["step"] -= 1
        st.rerun()


def render_progress():
    step = st.session_state["step"]
    st.progress((step - 1) / (len(STEP_NAMES) - 1))
    cols = st.columns(len(STEP_NAMES))
    for i, (col, name) in enumerate(zip(cols, STEP_NAMES), start=1):
        with col:
            if i < step:
                st.markdown(f"✅ {name}")
            elif i == step:
                st.markdown(f"**▶ {name}**")
            else:
                st.markdown(f"⬜ {name}")
    st.markdown("---")


def run_code(code: str, extra_vars: dict) -> dict:
    """
    학생이 편집한 코드를 실행한다.
    - pandas/numpy/sklearn/keras의 import 문이 정상 동작해야 하므로
      기본 builtins는 유지하되, 파일/OS 접근 관련 위험 함수만 제거한다.
    - 반환값: 실행 후 local 변수 딕셔너리 (오류 시 예외를 그대로 올림)
    """
    import builtins as _builtins

    safe_builtins = dict(vars(_builtins))
    for name in ("open", "exec", "eval", "compile", "input", "__import__"):
        # __import__는 sklearn/keras import에 필요하므로 제거하지 않고,
        # 대신 위험한 파일/실행 계열 함수만 제거한다.
        if name in ("open", "exec", "eval", "compile", "input"):
            safe_builtins.pop(name, None)

    scope = {"__builtins__": safe_builtins, "pd": pd, "np": np}
    scope.update(extra_vars)
    exec(code, scope)  # noqa: S102 - 교사가 검토/실행하는 교육용 로컬 앱 전제
    return scope


def code_box(key: str, recommended_code: str) -> str:
    """추천 코드를 보여주고 편집 가능한 text_area를 반환"""
    st.code(recommended_code, language="python")
    return st.text_area("코드 수정 후 실행하세요", value=recommended_code, height=180, key=f"code_{key}")


def show_named_series(series: pd.Series, index_label: str, value_label: str, max_width: int = 480):
    """
    컬럼명이 '0'처럼 무의미하게 나오는 Series 결과를,
    '컬럼명 / 값' 형태의 표로 바꾸고 화면 크기에 맞는 적당한 폭으로 표시한다.
    """
    display_df = series.rename(value_label).rename_axis(index_label).reset_index()
    st.dataframe(display_df, hide_index=True, width=max_width)


def show_dataframe_fit(df: pd.DataFrame, max_width: int = 700, height: int | None = None):
    """열 개수에 비례해 적당한 폭으로 데이터프레임을 표시 (화면 전체로 늘어나는 것 방지)"""
    n_cols = max(len(df.columns), 1)
    width = min(120 + n_cols * 110, max_width)
    if height is None:
        st.dataframe(df, width=width)
    else:
        st.dataframe(df, width=width, height=height)


def render_free_code_console(key_prefix: str = "console"):
    """
    학생이 직접 파이썬 코드를 입력하고 즉시 실행 결과를 확인할 수 있는 콘솔.
    df(현재 데이터), pd, np 를 기본으로 사용할 수 있다.
    - print() 출력, 마지막 줄의 값(자동 출력), DataFrame/Series/숫자/문자열 등을
      결과 형태에 맞게 알아서 보여준다.
    """
    import io
    import contextlib
    import ast

    with st.expander("🖥️ 직접 코드 작성 & 실행", expanded=False):
        st.caption("`df`(현재 데이터), `pd`, `np`를 바로 사용할 수 있습니다. 마지막 줄의 값이나 print() 출력이 아래에 표시됩니다.")
        code = st.text_area(
            "Python 코드 입력",
            value="df.head()",
            height=160,
            key=f"{key_prefix}_free_code",
        )

        if st.button("▶ 실행", key=f"{key_prefix}_free_run", type="primary"):
            df = st.session_state.get("df")
            extra_vars = {"df": df} if df is not None else {}
            stdout_buf = io.StringIO()

            try:
                # 마지막 줄이 단순 표현식(예: df.head())이면 결과를 자동으로 잡아서 보여준다.
                tree = ast.parse(code, mode="exec")
                last_expr_node = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr_node = tree.body.pop()

                body_src = ast.unparse(tree) if tree.body else ""
                if last_expr_node is not None:
                    result_line = f"_console_result_ = {ast.unparse(last_expr_node.value)}"
                    full_code = f"{body_src}\n{result_line}" if body_src else result_line
                else:
                    full_code = body_src

                with contextlib.redirect_stdout(stdout_buf):
                    scope = run_code(full_code, extra_vars)

                printed = stdout_buf.getvalue()
                if printed:
                    st.text(printed)

                result = scope.get("_console_result_")
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        show_dataframe_fit(result)
                    elif isinstance(result, pd.Series):
                        show_dataframe_fit(result.to_frame(name="값"))
                    else:
                        st.write(result)
                elif not printed:
                    st.info("코드는 정상 실행됐지만 화면에 표시할 출력이 없습니다. (print()를 쓰거나 마지막 줄에 값을 두세요)")
            except Exception as e:
                st.error(f"실행 오류: {e}")
