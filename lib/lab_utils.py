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
