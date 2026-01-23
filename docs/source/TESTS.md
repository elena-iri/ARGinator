# Testing Documentation

This project uses `pytest` for unit and integration testing. The tests are divided into three categories: Data, Model Architecture, and Scripts (Training/Evaluation).

## 1. Data Tests (`tests/test_data.py`)
*Tests the data ingestion pipeline, H5 file parsing, and DataLoader construction.*

| Test Function | Description |
| :--- | :--- |
| `test_get_dataloaders` | **Integration Test.** Creates temporary dummy `.h5` files with specific naming conventions ("non" vs "protein") to verify that: <br> 1. The `Dataset` class parses H5 keys correctly. <br> 2. Labels are assigned correctly (0 or 1) based on filenames. <br> 3. `DataLoader` collates individual samples into batches of correct shape `[Batch_Size, 1024]`. |

## 2. Model Tests (`tests/test_model.py`)
*Tests the Neural Network architecture and mathematical correctness.*

| Test Function | Description |
| :--- | :--- |
| `test_model_forward_shape` | **Parametrized Unit Test.** Runs the model with batch sizes `[1, 32, 64]`. Verifies that input tensor `[N, 1024]` results in output `[N, 2]` without crashing or producing NaNs. |
| `test_model_backward` | **Smoke Test.** Runs a full forward and backward pass on dummy data to ensure gradients are calculated. Catches errors like broken computational graphs or detached tensors. |
| `test_model_dropout_initialization` | **Unit Test.** Verifies the model can be initialized with different dropout rates (0.0 vs 0.5) without error. |

## 3. Script Tests (`tests/test_train.py`, `tests/test_evaluate.py`)
*Tests the orchestration logic, CLI arguments, and file saving. These tests use Mocks to avoid running heavy computations.*

| Test Function | Description |
| :--- | :--- |
| `test_train` | **Mocked Logic Test.** bypasses the `@hydra.main` decorator to test the training loop. <br> - **Mocks:** `HydraConfig`, `plt`, `get_dataloaders`. <br> - **Verifies:** The training loop runs, the model file (`.pth`) is saved to disk, and the plotting function (`plt.savefig`) is triggered. |
| `test_evaluate` | **Mocked Logic Test.** Tests the evaluation script. <br> - **Mocks:** `Model`, `torch.load`. <br> - **Verifies:** The script loads the config, reloads weights from disk, enters `eval()` mode, and correctly calculates accuracy on the test set. |

## How to Run
To run all tests:
```bash
uv run pytest tests/
```

To run a specific category (e.g., only model tests):
```bash
uv run pytest tests/test_model.py
```
**4\. Code Coverage Report**

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
