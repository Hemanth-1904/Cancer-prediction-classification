import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ml_tools import run_ML_pipeline
from xg_ml import run_XG_pipeline
import io

st.set_page_config(page_title="Cancer Prediction and Classification App", layout="centered")

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'file_path' not in st.session_state:
    st.session_state.file_path = None
if 'algorithm' not in st.session_state:
    st.session_state.algorithm = None

def reset_session():
    st.session_state.step = 1
    st.session_state.uploaded_file = None
    st.session_state.file_path = None
    st.session_state.algorithm = None

def input_window():
    st.markdown("<h1 style='text-align: center;'>Cancer Prediction and Classification App</h1>", unsafe_allow_html=True)
    
    algorithm = st.radio(
        "Select Algorithm",
        ["Random Forest", "XGBoost"],
        key="algorithm_selection"
    )
    st.session_state.algorithm = "RF" if algorithm == "Random Forest" else "XG"
    
    st.markdown(f"<h3 style='text-align: center;'>Analysis using {algorithm}</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload the dataset (should be .csv/.tsv file, the first column must have Gene_ID)", type=["csv", "tsv"])

    if uploaded_file:
        try:
            st.session_state.uploaded_file = uploaded_file
            file_path = "temp_" + uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.file_path = file_path

            # Preview the uploaded data
            df = pd.read_csv(file_path, index_col=0)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            if st.button("Run Analysis"):
                st.session_state.step = 2
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def result_window():
    if st.button("⬅️ Back to Input Window"):
        reset_session()
        st.rerun()

    st.markdown("<h1 style='text-align: center;'>Results</h1>", unsafe_allow_html=True)

    try:
        if st.session_state.uploaded_file and st.session_state.file_path:
            pipeline_func = run_ML_pipeline if st.session_state.algorithm == "RF" else run_XG_pipeline

            with st.sidebar:
                st.markdown("<h3>Reports</h3>", unsafe_allow_html=True)
                report_type = st.radio(
                    label="Choose a report to view",
                    options=["Prediction Result", "Confusion Matrix", "AUC Curve"],
                    index=0
                )

            if report_type == "Prediction Result":
                with st.spinner("Generating prediction results..."):
                    result = pipeline_func("prediction_result", st.session_state.file_path, st.session_state.algorithm)
                    st.markdown("<h2 style='text-align: center;'>Prediction Results</h2>", unsafe_allow_html=True)
                    st.dataframe(result, width=900)
                    
                    csv = result.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Prediction Result",
                        data=csv,
                        file_name='prediction_results.csv',
                        mime='text/csv'
                    )

            elif report_type == "Confusion Matrix":
                with st.spinner("Generating confusion matrix..."):
                    confusion_matrix_figure = pipeline_func("confusion_matrix", st.session_state.file_path, st.session_state.algorithm)
                    st.markdown("<h2 style='text-align: center;'>Confusion Matrix</h2>", unsafe_allow_html=True)
                    st.pyplot(confusion_matrix_figure)
                    
                    buf = io.BytesIO()
                    confusion_matrix_figure.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="Download Confusion Matrix",
                        data=buf,
                        file_name='confusion_matrix.png',
                        mime="image/png"
                    )

            elif report_type == "AUC Curve":
                with st.spinner("Generating AUC curve..."):
                    auc_curve_figure = pipeline_func("roc_auc_curve", st.session_state.file_path, st.session_state.algorithm)
                    st.markdown("<h2 style='text-align: center;'>AUC Curve</h2>", unsafe_allow_html=True)
                    st.pyplot(auc_curve_figure)
                    
                    buf = io.BytesIO()
                    auc_curve_figure.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="Download AUC Curve",
                        data=buf,
                        file_name='auc_curve.png',
                        mime="image/png"
                    )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.button("Reset Application"):
            reset_session()
            st.rerun()

def main():
    if st.session_state.step == 1:
        input_window()
    elif st.session_state.step == 2:
        result_window()

if __name__ == "__main__":
    main()