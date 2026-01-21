# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [✅] Create a git repository (M5)
* [✅] Make sure that all team members have write access to the GitHub repository (M5)
* [✅] Create a dedicated environment for you project to keep track of your packages (M2)
* [✅] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [✅] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [✅] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [✅] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7) TOMS **DO THIS AT THE VERY END**
* [✅] Do a bit of code typing and remember to document essential parts of your code (M7)
* [✅] Setup version control for your data or part of your data (M8) **optional**
* [✅] Add command line interfaces and project commands to your code where it makes sense (M9)
* [✅] Construct one or multiple docker files for your code (M10) 
* [✅] Build the docker files locally and make sure they work as intended (M10)
* [✅] Write one or multiple configurations files for your experiments (M11) 
* [✅] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12) - TOMS
* [✅] Use logging to log important events in your code (M14) MAX 
* [✅] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14) Elena 
* [✅] Consider running a hyperparameter optimization sweep (M14) Elena
      https://wandb.ai/eleni-iriondo2-danmarks-tekniske-universitet-dtu/arginator_protein_classifier/reports/Hyperparameter-sweep--VmlldzoxNTYxMzY1Mw
* [✅] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15) Optional Claudio
* [✅] Revise cookicutter struct

### Week 2

* [✅] Write unit tests related to the data part of your code (M16) - MAX
* [✅] Write unit tests related to model construction and or model training (M16) - MAX
* [✅] Calculate the code coverage (M16) - MAX
* [✅] Get some continuous integration running on the GitHub repository (M17) Max
* [✅] Add caching and multi-os/python/pytorch testing to your continuous integration (M17) Max
* [✅] Add a linting step to your continuous integration (M17) Elena 
* [✅] Add pre-commit hooks to your version control setup (M18) Elena 
* [✅] Add a continues workflow that triggers when data changes (M19) Free 
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19) Elena 
* [✅] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21) Claudio 
* [ ] Create a trigger workflow for automatically building your docker images (M21) Claudio 
* [✅] Get your model training in GCP using either the Engine or Vertex AI (M21) Claudio 
* [✅] Create a FastAPI application that can do inference using your model (M22) Toms
* [✅] Deploy your model in GCP using either Functions or Run as the backend (M23) Toms 
* [✅] Write API tests for your application and setup continues integration for these (M24) Toms 
* [✅] Load test your application (M24) - Toms 
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25) **optional** 
* [✅] Create a frontend for your API (M26) Toms

### Week 3

* [✅] Check how robust your model is towards data drifting (M27) Elena
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [✅] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 16

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s243312, s215141, s253510, s215145

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

(It was covered by the course but) we used the Pytorch-Lightning framework to reduce boilerplate ML code in our codebase. A python package that we used outside of the course was the h5py library to process .h5 protein embedding files into a torch tensor. We also employed umap to generate a figure in our API when running inference. Lastly, we used the ProtT5 protein languague model from hugging face model to generate the input data (protein embeddings) for our classifier.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used uv for managing our dependencies and the Python environment. Our direct list of dependencies was auto-generated using uv and are declared in the `pyproject.toml` file, while `uv` generates and maintains the cross-platform `uv.lock` file. To get a complete copy of our development environment, first clone this repo, next run the command `uv sync`, and subsequently run the different scripts in the src folder with `uv run script.py`. We also implemented DVC to manage large files so to pull the most up to date data run `uv run dvc pull`.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We closely followed the structure of the cookiecutter template. We filled out the configs, dockerfiles, tests and src folders. We have added a folder for outputs and for wandb logging (the latter is .gitignored) and removed the reports folder as this was redundant for us. 

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We used the ruff libary for code linting and formatting. We also used Typer for adding the CLI commands (to e.g. `uv run train` instead of `uv run train.py`). These concepts are important in larger projects to ensure that code is clean and consistent making it possible for all team members to easily follow what is going on in the code, which is especially useful for the case that they want to continue the work on something.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we implemented 4 tests. We are primarily testing the train, evaluate, data processing and api scripts as those are the most critical in our application. The tests and code coverage are documented in the `TESTS.md` file in the tests folder.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

*Generated via `pytest-cov`*

The overall project test coverage is **68%**. Below is the detailed breakdown by module:

| File | Statements | Missed | Coverage | Missing Lines |
| :--- | :---: | :---: | :---: | :--- |
| `data.py` | 171 | 72 | **58%** | 37, 47, ... , 298-329 |
| `model.py` | 88 | 30 | **66%** | 61-62, 86-105, 108-119 |
| `train.py` | 76 | 4 | **95%** | 109-110, 139, 174 |
| `__init__.py` | 0 | 0 | **100%** | - |
| **TOTAL** | **335** | **106** | **68%** | |

*Analysis of Missing Lines*
* **`data.py` (58%)**: The coverage is lower because `test_data.py` focuses on the **Binary** task flow.
    * **Missing Logic:** The `multiclass` labeling logic in `_labeling` (lines 91-102) and specific error handling branches (e.g., file read errors) are not triggered.
    * **Hydra Entry Point:** The `main()` function and CLI entry block (lines 298-334) are not executed by the unit tests.
* **`model.py` (66%)**:
    * **Lightning Methods:** The tests check `forward` and `backward`, but they do not execute `training_step` (86-105) or `validation_step` (108-119). These methods are usually called by the Lightning Trainer, which we mocked in `test_train.py`. To increase this, we would need to manually call `model.training_step()` in a unit test.
* **`train.py` (95%)**: High coverage. The few missing lines correspond to specific `multiclass` ROC plotting branches (since the test used binary data) or specific fallback logic for dataloaders.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Yes, we made use of both branches and PRs in our project. For each new feature (checklist item) that we implemented we would create a new branch e.g. `unit_tests`, `data-version-control`, `gcp_cloud` and then merge these, via pull requests, into a `development` branch. If we were then happy with a version of the code and everything seemed to be working properly we would merge this `development` branch into `main` (also via PR). For the Pull Requests we tried to always wait for at least one other group member to have a look and review before merging the branches. This setup allowed for revisiting certain features when they no longer worked with the current code due to other conflicting features. An example of this would be the `unit_tests` which we had to update after we changed certain parts of our data processing and training code with torch-lightning.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

Yes, DVC was heavily utilized with GCP in our project as our remote backend to manage the large protein embedding files.

It improved our project primarily by decoupling the code from the data while maintaining a strict link between them. Instead of bloating the Git repository with gigabytes of .h5 files, we instead committed small .dvc pointer files. This ensured that every commit in our history referenced the exact version of the dataset used at that time, making experiments fully reproducible. If a model performance dropped, we could checkout the previous Git commit and run `uv run dvc pull` to instantly revert the data state to match with the code.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We organized our Continuous Integration (CI) into four workflows to efficiently handle different parts of the pipeline:

1. Code Quality & Linting: We use two workflows, `Code linting` and `Pre-commit CI`. These run ruff to enforce coding standards and formatting. The pre-commit workflow is particularly robust; it can automatically fix minor style issues and push those changes back to the branch, reducing manual interventions.
2. Unit Testing: Our `Unit Tests` workflow is designed for maximum compatibility. We utilize a build matrix strategy to test our code across three operating systems: Ubuntu, Windows, and macOS, two Python versions: 3.11, 3.12, and two different PyTorch versions: 2.4.0, 2.5.1. This ensures that the package is robust and platform-agnostic.
3. Data Integrity: Uniquely, we implement a DVC Workflow that triggers specifically on changes to .dvc files. It pulls the data from Google Cloud Storage, runs a custom validation script, and uses CML to post a visual data quality report as a comment directly to the Pull Requests.

We make extensive use of caching for optimizing runtime. Across all workflows, we use astral-sh/setup-uv with enable-cache: true to cache our Python dependencies. Additionally, in the data workflow, we explicitly cache the .dvc/cache directory, which significantly speeds up the downloading of large protein embedding files from the cloud.

An example of our testing workflow can be seen here: [https://github.com/elena-iri/ARGinator/actions/runs/21096186032]

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used separate config files for experiment, optimizer, paths, processing and the task (binary/multiclass), based on which task was then specified in the `train_config.yaml` the corresponding `output_dim` would be loaded from the task config folder. We used Hydra for keeping config log files of the experiment, this slightly changed the default double hyphen structure from typer to use a full stop for custom specifications e.g. `uv run train experiment.lr=1e-2`.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We ensured reproducibility by tightly coupling the code, data, and configuration using a combination of Hydra, WandB, and DVC.
We used Hydra to manage complex, hierarchical configurations, ensuring that all hyperparameters are defined explicitly in code rather than hardcoded. whenever an experiment is run, the following happens:

- WandB automatically captures the fully resolved Hydra configuration (config.yaml), ensuring we know the exact parameters used, even if command-line overrides were applied.
- It logs the Git commit hash, which locks the version of the code.
- Because we are using DVC, that specific Git commit also points to the exact version of the dataset used during training.

To reproduce a past experiment, one would simply have to checkout the specific git commit logged in WandB (restoring both code and DVC data pointers) and execute the training script using the saved config file from the WandB dashboard. This guarantees that the exact code, data and parameters are reproduced.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

<img width="773" height="650" alt="image" src="https://github.com/user-attachments/assets/3b816abb-eb8b-496b-b765-cb28e79200b6" />

As seen in the above image we are tracking the basic training metrics such as the loss, recall and precision during a training run that inform us whether or not the model is improving over epochs.

<img width="640" height="480" alt="media_images_roc_curve_10_d2d2d0cc80e2366999c7" src="https://github.com/user-attachments/assets/a37a2ecd-3d25-4851-84ce-70eec32fa365" />

We also create and store Receiver Operating Characteristic (ROC) curves using matplotlib at the end of training and upload these as a .png to WandB, to understand if the model is just randomly guessing or making reasonable classifications. 

<img width="874" height="586" alt="image" src="https://github.com/user-attachments/assets/806fa7e1-29de-49fc-8023-71c656f4fa13" />

We also created a hyperparameter sweep to see which hyperparameters most significantly affect the validation loss, we saw that the most drastic differences are seen only when the batch size is lowered but that other parameters (in the ranges we tested) did not effect the loss to as large an extent.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 24 fill here ---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We implemented monitoring by testing for data drift using the framework Evidently. 

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

