# Testing Documentation

This project uses `pytest` for unit and integration testing. The tests are divided into three categories: Data (PyTorch Datasets), Model Architecture (LightningModules), and Scripts (Training/Orchestration).

## 1. Data Tests (`tests/test_data.py`)
*Tests the data ingestion pipeline, H5 file parsing, caching logic, and Stratified Splitting.*

| Test Function | Description |
| :--- | :--- |
| `test_get_dataloaders` | **Integration Test.** Uses `monkeypatch` to mock environment variables (`FILE_PATTERN`, `CLASSES`) and creates dummy `.h5` files to verify: <br> 1. `MyDataset` correctly parses files and caches them to a `.pt` file. <br> 2. The stratified split logic correctly divides data into Train/Val/Test based on ratios. <br> 3. `DataLoader` collates individual samples into batches of shape `[Batch_Size, 1024]`. <br> 4. Class balance is preserved in the training subset. |

## 2. Model Tests (`tests/test_model.py`)
*Tests the `Lightning_Model` architecture, configuration injection, and gradient flow.*

| Test Function | Description |
| :--- | :--- |
| `test_model_forward_shape` | **Parametrized Unit Test.** Runs the `Lightning_Model` with batch sizes `[1, 32, 64]`. Verifies that input tensor `[N, 1024]` results in output `[N, Output_Dim]` without producing NaNs. |
| `test_model_backward` | **Smoke Test.** Runs a forward pass and manually triggers `loss.backward()` to ensure the computational graph is connected and gradients are generated for parameters. |
| `test_model_initialization` | **Unit Test.** Uses `pytest` fixtures to inject mock Hydra configs (Loss/Optimizer). Verifies the model initializes correctly with different hyperparameters (e.g., Dropout rates). |
| `test_optimizer_configuration`| **Configuration Test.** Verifies that `configure_optimizers` successfully instantiates a valid PyTorch optimizer from the Hydra config. |

## 3. Script Tests (`tests/test_train.py`)
*Tests the training orchestration, PyTorch Lightning Trainer integration, and logging. Uses extensive mocking to avoid heavy computation.*

| Test Function | Description |
| :--- | :--- |
| `test_train` | **Mocked Logic Test.** Bypasses the `@hydra.main` decorator to test the full training loop. <br> - **Mocks:** `HydraConfig`, `TL_Dataset`, `Trainer` (Lightning), `WandB`, and `WandbLogger`. <br> - **Verifies:** <br> 1. The Lightning `Trainer.fit()` is called with the correct model and data. <br> 2. The `evaluate()` loop runs successfully on mock data. <br> 3. WandB summaries are updated with test metrics. |

## How to Run

To run all tests:

    uv run pytest tests/

To run a specific category (e.g., only model tests):

    uv run pytest tests/test_model.py

## 4. Code Coverage Report
*Generated via `pytest-cov`*

The overall project test coverage is **68%**. Below is the detailed breakdown by module:

| File | Statements | Missed | Coverage | Missing Lines |
| :--- | :---: | :---: | :---: | :--- |
| `data.py` | 171 | 72 | **58%** | 37, 47, ... , 298-329 |
| `model.py` | 88 | 30 | **66%** | 61-62, 86-105, 108-119 |
| `train.py` | 76 | 4 | **95%** | 109-110, 139, 174 |
| `__init__.py` | 0 | 0 | **100%** | - |
| **TOTAL** | **335** | **106** | **68%** | |

### Analysis of Missing Lines
* **`data.py` (58%)**: The coverage is lower because `test_data.py` focuses on the **Binary** task flow.
    * **Missing Logic:** The `multiclass` labeling logic in `_labeling` (lines 91-102) and specific error handling branches (e.g., file read errors) are not triggered.
    * **Hydra Entry Point:** The `main()` function and CLI entry block (lines 298-334) are not executed by the unit tests.
* **`model.py` (66%)**:
    * **Lightning Methods:** The tests check `forward` and `backward`, but they do not execute `training_step` (86-105) or `validation_step` (108-119). These methods are usually called by the Lightning Trainer, which we mocked in `test_train.py`. To increase this, we would need to manually call `model.training_step()` in a unit test.
* **`train.py` (95%)**: High coverage. The few missing lines correspond to specific `multiclass` ROC plotting branches (since the test used binary data) or specific fallback logic for dataloaders.