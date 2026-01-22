import sys
import os
import shutil
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# 1. SETUP PATHS (So imports work)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 2. IMPORT APP
# Ensure this matches your folder structure exactly
from src.arginator_protein_classifier.backend import app, JOBS 

# 3. DEFINE THE PATH STRING 
# We save this string to a variable so we don't have to type it 5 times
MODULE_PATH = "src.arginator_protein_classifier.backend"

@pytest.fixture
def mock_env(tmp_path):
    # ... (Keep your existing mock_env code exactly the same) ...
    inference_dir = tmp_path / "inference"
    binary_model_dir = tmp_path / "binary_models"
    multiclass_model_dir = tmp_path / "multiclass_models"
    inference_dir.mkdir()
    binary_model_dir.mkdir()
    multiclass_model_dir.mkdir()
    (binary_model_dir / "model.ckpt").touch()
    (multiclass_model_dir / "model.ckpt").touch()

    mock_cfg = MagicMock()
    mock_cfg.paths.t5_model_dir = str(tmp_path / "t5_model")
    mock_cfg.paths.data_inference_dir = str(inference_dir)
    mock_cfg.paths.binary_model_dir = str(binary_model_dir)
    mock_cfg.paths.multiclass_model_dir = str(multiclass_model_dir)
    mock_cfg.paths.card_A = str(tmp_path / "card_A")
    mock_cfg.paths.card_B = str(tmp_path / "card_B")
    mock_cfg.paths.card_C = str(tmp_path / "card_C")
    mock_cfg.paths.card_D = str(tmp_path / "card_D")
    return mock_cfg, inference_dir

def test_full_job_workflow(mock_env):
    mock_cfg_obj, inference_dir = mock_env

    # --- CRITICAL FIX HERE ---
    # Use f-strings with MODULE_PATH to ensure patch finds the file
    with patch(f"{MODULE_PATH}.load_t5_model", return_value=("MockModel", "MockVocab")) as mock_load, \
         patch(f"{MODULE_PATH}.initialize") as mock_init, \
         patch(f"{MODULE_PATH}.compose", return_value=mock_cfg_obj) as mock_compose, \
         patch(f"{MODULE_PATH}.run_conversion") as mock_convert, \
         patch(f"{MODULE_PATH}.run_inference") as mock_inference, \
         patch(f"{MODULE_PATH}.UMAPEmbeddingVisualizer") as mock_umap_cls:

        # ... (Rest of the test logic remains the same) ...
        
        # Mock side effects
        def side_effect_inference(checkpoint_path, data_path, output_dir, job_id):
            csv_path = os.path.join(output_dir, f"{job_id}_results.csv")
            df = pd.DataFrame({"protein": ["Seq1"], "score": [0.95]})
            df.to_csv(csv_path, index=False)
        mock_inference.side_effect = side_effect_inference

        mock_visualizer_instance = mock_umap_cls.return_value
        def side_effect_umap(file_map, binary_mode, output_filename):
            with open(output_filename, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n") 
        mock_visualizer_instance.run.side_effect = side_effect_umap

        with TestClient(app) as client:
            # 1. Submit
            file_content = b">seq1\nMKTLLLINVALID"
            files = {'file': ('test.fasta', file_content, 'text/plain')}
            form_data = {'classification_type': 'Binary'}
            
            response = client.post("/submit_job", files=files, data=form_data)
            assert response.status_code == 200, response.text
            job_id = response.json()["job_id"]

            # 2. Status
            status_resp = client.get(f"/status/{job_id}")
            assert status_resp.status_code == 200
            assert status_resp.json()["status"] == "completed"

            # 3. Download
            download_resp = client.get(f"/download/{job_id}")
            assert download_resp.status_code == 200