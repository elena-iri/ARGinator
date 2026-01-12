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
## 4. Code Coverage Report
*Generated via `pytest-cov`*

The overall project test coverage is **88%**. Below is the detailed breakdown by module:

| File | Statements | Missed | Coverage | Missing Lines |
| :--- | :---: | :---: | :---: | :--- |
| `data.py` | 73 | 13 | **82%** | 44-45, 57-58, 101-116, 119 |
| `evaluate.py` | 38 | 6 | **84%** | 39-43, 61 |
| `model.py` | 28 | 4 | **86%** | 29-32 |
| `train.py` | 55 | 1 | **98%** | 97 |
| `__init__.py` | 0 | 0 | **100%** | - |
| **TOTAL** | **194** | **24** | **88%** | |

### Analysis of Missing Lines
* **`data.py` (82%)**: The missing lines likely correspond to the main execution block (`if __name__ == "__main__":`) or specific error handling branches (e.g., `except Exception as e`) that were not triggered during the integration test.
* **`evaluate.py` (84%)**: The missing block (lines 39-43) is the `FileNotFoundError` handling logic, which we did not explicitly trigger in the success-case test.
* **`model.py` (86%)**: The missing lines (29-32) are likely the `if __name__ == "__main__":` block or debug print statements.
