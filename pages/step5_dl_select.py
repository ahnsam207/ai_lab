import streamlit as st
from lab_utils import go_next, go_prev


def render():
    st.subheader("5단계 — 딥러닝 선택")

    df = st.session_state["df"]
    if df is None:
        st.warning("이전 단계를 먼저 완료해주세요.")
        return

    st.caption(
        "현재 파이프라인은 표(정형) 데이터를 다루므로 CNN(이미지)·RNN(시계열/순서형)은 "
        "이 실습에서 지원하지 않습니다. 이미지·시계열 데이터를 다루려면 별도 입력 단계가 필요합니다."
    )
    arch = st.radio(
        "신경망 구조 (필수)",
        ["MLP (다층퍼셉트론)", "CNN (미지원 - 이미지 데이터 전용)", "RNN (미지원 - 시계열 데이터 전용)"],
    )
    if arch != "MLP (다층퍼셉트론)":
        st.warning("현재 데이터로는 이 구조를 실행할 수 없습니다. 'MLP'를 선택해주세요.")

    layers = st.radio("레이어 수 (필수)", ["2층", "3층", "4층"])

    st.session_state["dl_arch"] = arch
    st.session_state["dl_layers"] = {"2층": 2, "3층": 3, "4층": 4}[layers]

    if arch == "MLP (다층퍼셉트론)":
        n = st.session_state["dl_layers"]
        code_preview = "from tensorflow import keras\nmodel = keras.Sequential([\n"
        code_preview += "    keras.layers.Input(shape=(X_train.shape[1],)),\n"
        for _ in range(n - 1):
            code_preview += "    keras.layers.Dense(32, activation='relu'),\n"
        code_preview += "    keras.layers.Dense(1)  # 출력층: 문제 유형에 맞게 6단계에서 조정됨\n])"
        st.markdown("### 추천 코드 미리보기")
        st.code(code_preview, language="python")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← 이전 단계"):
            go_prev()
    with c2:
        if st.button("다음 단계 →", type="primary", disabled=(arch != "MLP (다층퍼셉트론)")):
            go_next()
