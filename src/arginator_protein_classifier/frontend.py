import streamlit as st
import requests
import time
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Arginator Protein Classifier", layout="centered")

st.sidebar.header("Configuration")
BACKEND_URL = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000")

st.title("🧬 Protein Classification System")
st.markdown("Upload a FASTA file to classify proteins using the T5 + Lightning model.")

uploaded_file = st.file_uploader("Upload .fa file", type=["fa", "fasta"])
class_type = st.selectbox("Choose Classification Type:", ["Binary", "Multiclass"])

if st.button("Run Inference", type="primary"):
    if uploaded_file is None:
        st.warning("!! Please upload a .fa file first.")
    else:
        try:
            # 1. SUBMIT JOB
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
            data = {"classification_type": class_type}
            
            with st.spinner("Uploading and starting job..."):
                submit_response = requests.post(f"{BACKEND_URL}/submit_job", files=files, data=data)
            
            if submit_response.status_code != 200:
                st.error(f"Failed to submit job: {submit_response.text}")
            else:
                job_id = submit_response.json().get("job_id")
                
                # 2. POLL FOR STATUS
                progress_bar = st.progress(0, text="Initializing processing...")
                status_container = st.empty()
                
                while True:
                    status_res = requests.get(f"{BACKEND_URL}/status/{job_id}")
                    
                    if status_res.status_code == 200:
                        job_data = status_res.json()
                        status = job_data.get("status")
                        progress = job_data.get("progress", 0)
                        
                        # Update Bar
                        progress_bar.progress(progress, text=f"Processing... {progress}%")
                        
                        # ... (inside the while True loop)

                        if status == "completed":
                            progress_bar.progress(100, text="Processing Complete!")
                            st.success("Job Finished Successfully")
                            
                            # --- 1. DISPLAY PLOT (NEW) ---
                            st.divider()
                            st.subheader("Visualization")
                            
                            # Request the plot from the new endpoint
                            plot_response = requests.get(f"{BACKEND_URL}/download_plot/{job_id}")
                            
                            if plot_response.status_code == 200:
                                st.image(
                                    plot_response.content, 
                                    caption="UMAP Projection of Protein Embeddings",
                                    width= 'stretch'
                                )
                            else:
                                st.warning("Visualization plot not available for this run.")

                            # --- 2. GET RESULTS & DOWNLOAD (EXISTING) ---
                            result = job_data.get("result", {})
                            
                            st.divider()
                            st.subheader("Prediction Results")
                            
                            if "preview" in result:
                                preview_df = pd.DataFrame(result["preview"])
                                st.write("Preview of first 5 rows:")
                                st.dataframe(preview_df, width='stretch')
                            
                            # Fetch the full CSV for download
                            download_res = requests.get(f"{BACKEND_URL}/download/{job_id}")
                            
                            if download_res.status_code == 200:
                                st.download_button(
                                    label="📥 Download Full Results (CSV)",
                                    data=download_res.content,
                                    file_name=f"results_{job_id}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("Could not fetch result file from server.")
                            
                            break
                    
                    time.sleep(1)

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to backend at `{BACKEND_URL}`. Is it running?")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")