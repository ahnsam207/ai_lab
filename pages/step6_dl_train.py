import streamlit as st
import pandas as pd
import numpy as np
from lab_utils import run_code, code_box, go_prev


def _build_code(n_layers, optimizer, problem_type, target_col, test_size,
                 epochs, batch_size, early_stop, dropout_rate):
    layer_lines = []
    for _ in range(n_layers - 1):
        layer_lines.append("    keras.layers.Dense(32, activation='relu'),")
        if dropout_rate:
            layer_lines.append(f"    keras.layers.Dropout({dropout_rate}),")

    if problem_type == "분류":
        out_layer = "keras.layers.Dense(n_classes, activation='softmax')"
        loss = "'sparse_categorical_crossentropy'"
        metrics = "['accuracy']"
    else:
        out_layer = "keras.layers.Dense(1)"
        loss = "'mse'"
        metrics = "['mae']"

    callbacks_setup = "callbacks = [st_callback]\n"
    if early_stop:
        callbacks_setup += "callbacks.append(keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True))\n"

    code = (
        "from tensorflow import keras\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import LabelEncoder\n\n"
        f"X = df.drop(columns=['{target_col}'])\n"
        f"y = df['{target_col}']\n"
    )
    if problem_type == "분류":
        code += "y = LabelEncoder().fit_transform(y)\nn_classes = len(set(y))\n"
    code += (
        f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)\n\n"
        "model = keras.Sequential([\n"
        "    keras.layers.Input(shape=(X_train.shape[1],)),\n"
        + "\n".join(layer_lines) + "\n"
        f"    {out_layer}\n"
        "])\n"
        f"model.compile(optimizer='{optimizer.lower()}', loss={loss}, metrics={metrics})\n\n"
        f"{callbacks_setup}"
        f"history = model.fit(X_train, y_train, validation_split=0.2, "
        f"epochs={epochs}, batch_size={batch_size}, callbacks=callbacks, verbose=0)\n"
        "test_result = model.evaluate(X_test, y_test, verbose=0)"
    )
    return code


def render():
    st.subheader("6단계 — 딥러닝 단계적 처리")

    df = st.session_state["df"]
    target_col = st.session_state["target_col"]
    problem_type = st.session_state["problem_type"]
    n_layers = st.session_state["dl_layers"]

    if df is None or n_layers is None:
        st.warning("이전 단계를 먼저 완료해주세요.")
        return
    if problem_type == "군집" or target_col is None:
        st.warning("딥러닝 학습은 분류/회귀 문제(타겟 컬럼 필요)에만 적용됩니다. 3단계에서 문제 유형을 다시 선택해주세요.")
        return

    optimizer = st.radio("옵티마이저 (필수)", ["Adam", "SGD", "RMSprop"])

    st.markdown("### 추가 설정 (선택)")
    do_epoch = st.checkbox("epoch 수 조정", value=True)
    epochs = st.slider("epoch 수", 5, 100, 20) if do_epoch else 20

    do_batch = st.checkbox("batch size 조정", value=True)
    batch_size = st.slider("batch size", 8, 128, 32, step=8) if do_batch else 32

    early_stop = st.checkbox("조기 종료 (EarlyStopping) 적용")

    do_dropout = st.checkbox("드롭아웃 적용")
    dropout_rate = st.slider("드롭아웃 비율", 0.0, 0.5, 0.2, step=0.05) if do_dropout else 0.0

    ratio_label = st.radio("train/test 분할 비율", ["70:30", "80:20", "90:10"], key="dl_split")
    test_size = {"70:30": 0.3, "80:20": 0.2, "90:10": 0.1}[ratio_label]

    code = code_box(
        "dl_train",
        _build_code(n_layers, optimizer, problem_type, target_col, test_size,
                    epochs, batch_size, early_stop, dropout_rate),
    )

    st.caption(
        "실행 코드 안의 `st_callback`은 아래에서 자동으로 주입되는 Streamlit 콜백입니다 "
        "(epoch마다 progress bar와 그래프를 갱신). tensorflow가 설치되어 있어야 실행됩니다."
    )

    if st.button("딥러닝 학습 실행", type="primary"):
        progress_bar = st.progress(0.0)
        chart_placeholder = st.empty()
        status_text = st.empty()
        history_log = {"loss": [], "val_loss": []}

        try:
            from tensorflow import keras

            class StreamlitCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    history_log["loss"].append(logs.get("loss"))
                    history_log["val_loss"].append(logs.get("val_loss"))
                    progress_bar.progress(min((epoch + 1) / epochs, 1.0))
                    status_text.text(f"epoch {epoch + 1}/{epochs} — loss: {logs.get('loss'):.4f}")
                    chart_placeholder.line_chart(pd.DataFrame(history_log))

            result = run_code(code, {
                "df": df, "pd": pd, "np": np,
                "st_callback": StreamlitCallback(), "epochs": epochs,
            })

            st.session_state["dl_model"] = result["model"]
            st.session_state["dl_history"] = result["history"].history
            st.success(f"학습 완료 — test 결과: {result['test_result']}")

            st.markdown("### 최종 결과 요약")
            st.json({
                "전처리 완료 데이터 shape": list(df.shape),
                "문제 유형": problem_type,
                "레이어 수": n_layers,
                "옵티마이저": optimizer,
                "epochs": epochs,
                "batch_size": batch_size,
                "조기종료": early_stop,
                "드롭아웃": dropout_rate,
                "test 평가결과": [float(x) for x in result["test_result"]],
            })
        except ImportError:
            st.error("tensorflow가 설치되어 있지 않습니다. `pip install tensorflow` 후 다시 시도해주세요.")
        except Exception as e:
            st.error(f"실행 오류: {e}")

    st.markdown("---")
    if st.button("← 이전 단계"):
        go_prev()
