**Testing Documentation**

This project uses pytest for unit and integration testing. The tests are divided into four categories: Data, Model Architecture, Training Scripts, and the FastAPI Backend.

## **1\. Data Tests (tests/test\_data.py)**

*Tests the data ingestion pipeline, H5 file parsing, and DataLoader construction.*

| Test Function | Description |
| :---- | :---- |
| test\_get\_dataloaders | **Integration Test.** Creates temporary dummy .h5 files with specific naming conventions ("non" vs "protein") to verify that: 1\. The Dataset class parses H5 keys correctly. 2\. Labels are assigned correctly (0 or 1\) based on filenames. 3\. DataLoader collates individual samples into batches of correct shape \[Batch\_Size, 1024\]. |

## **2\. Model Tests (tests/test\_model.py)**

*Tests the Neural Network architecture and mathematical correctness.*

| Test Function | Description |
| :---- | :---- |
| test\_model\_forward\_shape | **Parametrized Unit Test.** Runs the model with batch sizes \[1, 32, 64\]. Verifies that input tensor \[N, 1024\] results in output \[N, 2\] without crashing or producing NaNs. |
| test\_model\_backward | **Smoke Test.** Runs a full forward and backward pass on dummy data to ensure gradients are calculated. Catches errors like broken computational graphs or detached tensors. |
| test\_model\_dropout\_initialization | **Unit Test.** Verifies the model can be initialized with different dropout rates (0.0 vs 0.5) without error. |

## **3\. Script Tests (tests/test\_train.py)**

*Tests the orchestration logic, CLI arguments, and file saving. These tests use Mocks to avoid running heavy computations.*

| Test Function | Description |
| :---- | :---- |
| test\_train | **Mocked Logic Test.** Bypasses the @hydra.main decorator to test the training loop logic. \- **Mocks:** HydraConfig, plt, get\_dataloaders. \- **Verifies:** The training loop runs to completion, the model checkpoint (.pth) is saved to disk, and the loss curve plotting function is triggered. |

## **4\. API & Integration Tests (tests/integrationtests/test\_api.py)**

*Tests the FastAPI backend, background task processing, and endpoint responses. High-performance components (T5 Model, UMAP) are mocked to ensure fast execution.*

| Test Function | Description |
| :---- | :---- |
| test\_submit\_job\_endpoint | **Endpoint Test.** Verifies the /submit\_job endpoint. Checks that files are accepted, a Job ID is generated, the status is set to "processing," and a background task is scheduled. |
| test\_process\_file\_task\_success | **Logic Test.** Directly invokes the background worker function process\_file\_task. \- **Mocks:** run\_conversion, run\_inference, UMAPEmbeddingVisualizer. \- **Verifies:** The full workflow (Convert $\\to$ Infer $\\to$ Plot) runs sequentially, the job status updates to "completed," and temporary input files are cleaned up via os.remove. |
| test\_download\_endpoints | **Integration Test.** Creates real dummy files on disk to verify that /download/{job\_id} and /download\_plot/{job\_id} return the correct binary content and MIME types (text/csv, image/png). |
| test\_get\_status | **State Test.** Verifies that the status endpoint returns the correct JSON state ("processing", "completed", "failed") for a given Job ID. |

## How to Run
To run all tests:
```bash
uv run pytest tests/
```

To run a specific category (e.g., only model tests):
```bash
uv run pytest tests/test_model.py
```
## **5\. Code Coverage Report**

*Generated via pytest-cov on Windows (Python 3.12.10)*

The overall project test coverage is currently **38%**. While core backend and training logic is well-covered, several auxiliary and frontend modules remain untested. Below is the detailed breakdown by module:

| File | Statements | Missed | Coverage | Missing Lines |
| :---- | :---- | :---- | :---- | :---- |
| src\\arginator\_protein\_classifier\\backend.py | 121 | 18 | **85%** | 38, 60, 66-71, 93, 100, 112, 173-174, 201, 210, 223, 226, 233, 239 |
| src\\arginator\_protein\_classifier\\train.py | 99 | 9 | **91%** | 24-26, 34-35, 137-138, 188, 224 |
| src\\arginator\_protein\_classifier\\model.py | 87 | 32 | **63%** | 39-40, 64-88, 91-102, 114-115, 128-134, 138 |
| src\\arginator\_protein\_classifier\\data.py | 172 | 71 | **59%** | 44, 54, 56-58, 98-109, 120-121, 129, 146-147, 231-237, 241-268, 273, 276, 279, 288-320, 327 |
| src\\arginator\_protein\_classifier\\inference.py | 90 | 65 | **28%** | 24-33, 35, 37, 47-125, 143-146, 153 |
| src\\arginator\_protein\_classifier\\umap\_plot.py | 76 | 60 | **21%** | 22-29, 33-40, 48-52, 61-68, 72-79, 89-111, 122-133, 138-147 |
| src\\arginator\_protein\_classifier\\convertfa.py | 76 | 62 | **18%** | 18-55, 60-70, 78-136 |
| src\\arginator\_protein\_classifier\\data\_drift.py | 163 | 163 | **0%** | All lines (2-347) |
| src\\arginator\_protein\_classifier\\data\_validation.py | 116 | 116 | **0%** | All lines (1-175) |
| src\\arginator\_protein\_classifier\\frontend.py | 69 | 69 | **0%** | All lines (1-118) |
| src\\arginator\_protein\_classifier\\\_\_init\_\_.py | 0 | 0 | **100%** | \- |
| src\\arginator\_protein\_classifier\\visualize.py | 0 | 0 | **100%** | \- |
| **TOTAL** | **1069** | **665** | **38%** |  |

### **Analysis of Coverage**

* **High Coverage (backend.py: 85%, train.py: 91%)**: The critical API endpoints, job management logic, and training loops are thoroughly tested. Missing lines in backend.py are mostly error-handling branches (e.g., except Exception) that were not triggered during success-path testing.  
* **Moderate Coverage (model.py: 63%, data.py: 59%)**: The core model architecture and dataset logic are partially tested. The missing lines in model.py largely correspond to validation steps or specific property methods not used in the main training flow.  
* **Low/No Coverage**:  
  * data\_drift.py and data\_validation.py are currently completely untested (0%), likely because these are standalone monitoring scripts not invoked by the core API or training tests.  
  * frontend.py (0%) contains UI logic (Streamlit/Gradio), which requires a different testing strategy (e.g., Selenium or Playwright) not yet implemented in the pytest suite.  
  * inference.py and convertfa.py have low coverage because the tests mock their main functions (run\_inference, run\_conversion) to avoid heavy computation, leaving the internal logic of those functions unverified by the current test suite.
