import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import streamlit as st
from lab_utils import init_session, render_progress
import step1_input
import step2_preprocess
import step3_ml_select
import step4_ml_train
import step5_dl_select
import step6_dl_train

st.set_page_config(page_title="AI 학습 실습", layout="wide")
init_session()

st.title("🧪 AI 학습 실습 사이트")
render_progress()

STEP_RENDERERS = {
    1: step1_input.render,
    2: step2_preprocess.render,
    3: step3_ml_select.render,
    4: step4_ml_train.render,
    5: step5_dl_select.render,
    6: step6_dl_train.render,
}

STEP_RENDERERS[st.session_state["step"]]()
