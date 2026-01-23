import os
import shutil
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

# --- IMPORT APP ---
# Notice we import from the package name, not 'src'
from arginator_protein_classifier.backend import JOBS, app, process_file_task
from fastapi.testclient import TestClient

# --- FIXTURES ---

@pytest.fixture(scope="module")
def mock_config():
    """Mocks the Hydra Config object so we don't need real paths."""
    mock_cfg = MagicMock()
    # Define dummy paths that won't actually be used on disk because we mock file ops
    mock_cfg.paths.t5_model_dir = "dummy/t5"
    mock_cfg.paths.data_inference_dir = "dummy/inference"
    mock_cfg.paths.binary_model_dir = "dummy/binary_models"
    mock_cfg.paths.multiclass_model_dir = "dummy/multiclass_models"
    
    # Mock Card DB paths for UMAP
    mock_cfg.paths.card_A = "dummy/card_a.h5"
    mock_cfg.paths.card_B = "dummy/card_b.h5"
    mock_cfg.paths.card_C = "dummy/card_c.h5"
    mock_cfg.paths.card_D = "dummy/card_d.h5"
    return mock_cfg

@pytest.fixture(scope="module", autouse=True)
def mock_startup(mock_config):
    """
    Automatically mocks the 'lifespan' startup logic.
    This prevents loading the real T5 model or Hydra when creating the TestClient.
    """
    # --- FIX: Removed 'src.' from patch paths ---
    with patch("arginator_protein_classifier.backend.load_t5_model") as mock_load, \
         patch("arginator_protein_classifier.backend.initialize"), \
         patch("arginator_protein_classifier.backend.compose", return_value=mock_config):
        
        # Mock Model and Vocab
        mock_load.return_value = ("MockT5Model", "MockVocab")
        yield

@pytest.fixture
def client():
    """Creates a fresh TestClient for each test."""
    # The 'with' block triggers the lifespan startup/shutdown events
    with TestClient(app) as c:
        yield c

# --- ENDPOINT TESTS ---

def test_submit_job_endpoint(client, tmp_path):
    """Test the /submit_job endpoint."""
    
    # Mock os.makedirs and file writing so we don't clutter disk
    with patch("os.makedirs"), patch("builtins.open", mock_open()):
        
        # Prepare a dummy file upload
        file_content = b">seq1\nMVLSPADKTN..."
        files = {"file": ("test.fasta", file_content, "text/plain")}
        data = {"classification_type": "Binary"}

        # We mock the background task addition so the heavy logic doesn't actually run
        with patch("fastapi.BackgroundTasks.add_task") as mock_add_task:
            response = client.post("/submit_job", files=files, data=data)

    assert response.status_code == 200
    json_resp = response.json()
    
    assert "job_id" in json_resp
    assert json_resp["status"] == "submitted"
    
    # Verify the job was added to the global JOBS dict
    job_id = json_resp["job_id"]
    assert job_id in JOBS
    assert JOBS[job_id]["status"] == "processing"
    
    # Verify background task was scheduled
    mock_add_task.assert_called_once()

def test_get_status_endpoint(client):
    """Test /status/{job_id}"""
    # Manually inject a fake job
    fake_id = "123-fake-id"
    JOBS[fake_id] = {"status": "processing", "progress": 50}

    response = client.get(f"/status/{fake_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    assert response.json()["progress"] == 50

def test_get_status_not_found(client):
    response = client.get("/status/non-existent-id")
    assert response.status_code == 404

# --- LOGIC TESTS (The Background Task) ---

# --- FIX: Removed 'src.' from patch paths ---
@patch("arginator_protein_classifier.backend.run_conversion")
@patch("arginator_protein_classifier.backend.run_inference")
@patch("arginator_protein_classifier.backend.UMAPEmbeddingVisualizer")
@patch("os.remove")
@patch("os.path.exists")
@patch("pandas.read_csv")
@patch("arginator_protein_classifier.backend.CFG")
def test_process_file_task_success(
    mock_cfg, 
    mock_pd_read, 
    mock_exists, 
    mock_remove, 
    mock_umap_cls, 
    mock_inf, 
    mock_conv, 
    mock_config
):
    """
    Test the actual logic of process_file_task by calling it directly.
    """
    # Setup Global Config Mock
    mock_cfg.paths = mock_config.paths
    
    # Setup inputs
    job_id = "test_logic_job"
    JOBS[job_id] = {"status": "processing", "progress": 0}
    
    # Force exists() to return True so we pass the "model checkpoint exists" check
    mock_exists.return_value = True
    
    # Mock Pandas return for the preview
    mock_df = pd.DataFrame({"protein_id": ["A", "B"], "label": [0, 1]})
    mock_pd_read.return_value = mock_df

    # --- RUN THE FUNCTION ---
    process_file_task(job_id, "temp.fa", "temp.h5", "Binary")

    # --- ASSERTIONS ---
    
    # 1. Verify Job Status Updated
    assert JOBS[job_id]["status"] == "completed"
    assert JOBS[job_id]["progress"] == 100
    
    # 2. Verify Cleanup was attempted
    mock_remove.assert_called_once_with("temp.fa")

def test_process_file_task_failure():
    """Test that errors are caught and status is set to failed."""
    job_id = "fail_job"
    JOBS[job_id] = {"status": "processing"}
    
    # We call the function without mocking the dependencies properly, 
    # which will cause a crash (e.g. AttributeError because CFG is None or file not found)
    # forcing the 'except' block to run.
    process_file_task(job_id, "temp.fa", "temp.h5", "Binary")
    
    assert JOBS[job_id]["status"] == "failed"
    assert "error" in JOBS[job_id]

def test_download_endpoints(client, tmp_path):
    """
    Test downloading results and plots.
    """
    job_id = "download-test-job"
    
    # 1. Setup a "Completed" Job
    results_file = tmp_path / f"{job_id}_results.csv"
    plot_file = tmp_path / f"{job_id}_umap_plot.png"
    
    results_file.write_bytes(b"protein,score\nA,0.99")
    plot_file.write_bytes(b"fake_image_data")
    
    # 2. Inject this job into the Global JOBS dict
    JOBS[job_id] = {
        "status": "completed",
        "result": {
            "csv_path": str(results_file)
        }
    }

    # 3. Patch CFG without 'src.'
    with patch("arginator_protein_classifier.backend.CFG") as mock_cfg:
        mock_cfg.paths.data_inference_dir = str(tmp_path)

        # --- Test CSV Download ---
        response_csv = client.get(f"/download/{job_id}")
        assert response_csv.status_code == 200
        # Check if "text/csv" is IN the header (handles the extra "; charset=utf-8")
        assert "text/csv" in response_csv.headers["content-type"]
        assert response_csv.content == b"protein,score\nA,0.99"

        # --- Test Plot Download ---
        response_plot = client.get(f"/download_plot/{job_id}")
        assert response_plot.status_code == 200
        assert response_plot.headers["content-type"] == "image/png"
        assert response_plot.content == b"fake_image_data"

def test_download_not_ready(client):
    """Test downloading a job that is still processing."""
    job_id = "processing-job"
    JOBS[job_id] = {"status": "processing"}
    
    response = client.get(f"/download/{job_id}")
    assert response.status_code == 400 # Bad Request