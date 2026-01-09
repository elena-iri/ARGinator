# arginator_protein_classifier

Protein Language Model for classification as part of DTU MLOps course (02476) project by group 16.

## Project description
*   **Overall Goal:**

*   **Data to use:**
    We will use 1024-dimensional protein embeddings. These come from previous work by DTU researchers Alfred F. Florensa, Philip T. Lanken and DTU student Elena I. Delgado
*   **Models:**
    We currently use a fully connected Feed-Forward Neural Network (FFNN) which accepts pre-computed protein embeddings (1024-dim) as input. Initial experiments show promising results with this architecture. Since our data allows for multi-class or binary classfication tasks, our pipeline allows training to be conducted in any of the two aforementioned scenarios. Inference follows the same mindset. This results in us using two different models from a single one. Here the changes are seen in the number of neurons in the final layer, as well as the final activation function (sigmoid or Softmax)  

    To ensure robust and efficient performance, we plan to explore the following:

    Baselines: We will benchmark our Neural Network against classical linear models (e.g., Logistic Regression) to validate the necessity of a deep learning approach.

    Hyperparameter Search: We intend to systematically explore architecture variations (depth, width, dropout rates) to maximize accuracy without overfitting. This will be done with Weights And Biases (WandB)

    Efficiency Optimization: Given the potential deployment constraints, we plan to investigate model compression techniques like quantization or pruning to reduce inference latency and model size without significantly sacrificing accuracy. Additionally, this efficiency will be further explored with Pytorch Lightning as a way to generalize certain parts of our pipeline such as training modules. 
The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

