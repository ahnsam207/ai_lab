import streamlit as st
import pandas as pd
import numpy as np
from lab_utils import run_code, code_box, go_next, go_prev, show_dataframe_fit

SPLIT_RATIOS = {"70:30": 0.3, "80:20": 0.2, "90:10": 0.1}

TREE_ALGOS = {"랜덤포레스트", "랜덤포레스트회귀"}


def _train_code(algorithm, base_model_code, test_size, target_col, problem_type):
    return (
        f"{base_model_code}\n"
        "from sklearn.model_selection import train_test_split\n"
        f"X = df.drop(columns=['{target_col}'])\n"
        f"y = df['{target_col}']\n"
        f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)\n"
        "model.fit(X_train, y_train)\n"
        "y_pred = model.predict(X_test)"
    )


def _cluster_code(base_model_code):
    return (
        f"{base_model_code}\n"
        "X = df.select_dtypes(include='number')\n"
        "labels = model.fit_predict(X)"
    )


def render():
    st.subheader("4단계 — 머신러닝 단계적 처리")

    df = st.session_state["df"]
    problem_type = st.session_state["problem_type"]
    algorithm = st.session_state["algorithm"]
    target_col = st.session_state["target_col"]

    if df is None or algorithm is None:
        st.warning("이전 단계를 먼저 완료해주세요.")
        return

    from step3_ml_select import BASE_CODE
    base_model_code = BASE_CODE[algorithm]

    if problem_type == "군집":
        st.info("군집 분석은 train/test 분할이 필요 없습니다.")
        code = code_box("cluster_train", _cluster_code(base_model_code))
        if st.button("군집 학습 실행", type="primary"):
            try:
                result = run_code(code, {"df": df, "pd": pd, "np": np})
                labels = result["labels"]
                st.session_state["model"] = result["model"]
                st.session_state["ml_metrics"] = {"cluster_counts": pd.Series(labels).value_counts().to_dict()}
                st.success("군집 학습 완료")
                st.bar_chart(pd.Series(labels).value_counts().sort_index())
            except Exception as e:
                st.error(f"실행 오류: {e}")
    else:
        ratio_label = st.radio("train/test 분할 비율 (필수)", list(SPLIT_RATIOS.keys()))
        test_size = SPLIT_RATIOS[ratio_label]

        code = code_box(
            "ml_train",
            _train_code(algorithm, base_model_code, test_size, target_col, problem_type),
        )
        if st.button("학습 실행", type="primary"):
            try:
                result = run_code(code, {"df": df, "pd": pd, "np": np})
                st.session_state["model"] = result["model"]
                st.session_state["X_train"] = result["X_train"]
                st.session_state["X_test"] = result["X_test"]
                st.session_state["y_train"] = result["y_train"]
                st.session_state["y_test"] = result["y_test"]

                if problem_type == "분류":
                    from sklearn.metrics import accuracy_score, confusion_matrix
                    acc = accuracy_score(result["y_test"], result["y_pred"])
                    cm = confusion_matrix(result["y_test"], result["y_pred"])
                    st.session_state["ml_metrics"] = {"accuracy": acc}
                    st.success(f"정확도: {acc:.3f}")
                    st.write("혼동행렬 (행: 실제값, 열: 예측값)")
                    cm_df = pd.DataFrame(
                        cm,
                        index=[f"실제 {c}" for c in sorted(set(result["y_test"]))],
                        columns=[f"예측 {c}" for c in sorted(set(result["y_test"]))],
                    )
                    show_dataframe_fit(cm_df, max_width=500)
                else:
                    from sklearn.metrics import r2_score, mean_squared_error
                    r2 = r2_score(result["y_test"], result["y_pred"])
                    rmse = mean_squared_error(result["y_test"], result["y_pred"]) ** 0.5
                    st.session_state["ml_metrics"] = {"r2": r2, "rmse": rmse}
                    st.success(f"R²: {r2:.3f} · RMSE: {rmse:.3f}")
            except Exception as e:
                st.error(f"실행 오류: {e}")

        st.markdown("### 추가 처리 (선택)")
        do_cv = st.checkbox("교차검증 (cross_val_score)")
        do_tune = st.checkbox("하이퍼파라미터 튜닝 (GridSearchCV)")
        do_importance = st.checkbox(
            "특성 중요도 확인",
            disabled=algorithm not in TREE_ALGOS,
            help=None if algorithm in TREE_ALGOS else "트리 계열 모델(랜덤포레스트)에서만 지원합니다.",
        )

        if do_cv:
            folds = st.slider("fold 수", min_value=3, max_value=10, value=5)
            cv_code = (
                f"{base_model_code}\n"
                "from sklearn.model_selection import cross_val_score\n"
                f"X = df.drop(columns=['{target_col}'])\n"
                f"y = df['{target_col}']\n"
                f"scores = cross_val_score(model, X, y, cv={folds})"
            )
            cv_code = code_box("cv", cv_code)
            if st.button("교차검증 실행"):
                try:
                    result = run_code(cv_code, {"df": df, "pd": pd, "np": np})
                    st.success(f"평균 점수: {result['scores'].mean():.3f} (±{result['scores'].std():.3f})")
                    st.write(result["scores"])
                except Exception as e:
                    st.error(f"실행 오류: {e}")

        if do_tune:
            tune_code = (
                f"{base_model_code}\n"
                "from sklearn.model_selection import GridSearchCV, train_test_split\n"
                f"X = df.drop(columns=['{target_col}'])\n"
                f"y = df['{target_col}']\n"
                f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)\n"
                "param_grid = {'n_estimators': [50, 100, 200]} if hasattr(model, 'n_estimators') else {}\n"
                "grid = GridSearchCV(model, param_grid, cv=3) if param_grid else None\n"
                "if grid:\n"
                "    grid.fit(X_train, y_train)\n"
                "    best_params = grid.best_params_\n"
                "else:\n"
                "    model.fit(X_train, y_train)\n"
                "    best_params = '이 모델은 튜닝할 파라미터 그리드가 지정되지 않았습니다.'"
            )
            tune_code = code_box("tune", tune_code)
            if st.button("튜닝 실행"):
                try:
                    result = run_code(tune_code, {"df": df, "pd": pd, "np": np})
                    st.success(f"최적 파라미터: {result['best_params']}")
                except Exception as e:
                    st.error(f"실행 오류: {e}")

        if do_importance and algorithm in TREE_ALGOS:
            imp_code = code_box(
                "importance",
                "importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)",
            )
            if st.button("특성 중요도 실행"):
                model = st.session_state.get("model")
                X_train = st.session_state.get("X_train")
                if model is None or X_train is None:
                    st.warning("먼저 위에서 '학습 실행'을 완료해주세요.")
                else:
                    try:
                        result = run_code(imp_code, {"model": model, "X_train": X_train, "pd": pd, "np": np})
                        st.bar_chart(result["importances"])
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
