# PyTIR

This is the artifact of the research paper, "Co-Evolution of Types and Dependencies: Towards Repository-Level Type Inference for Python Code", submitted to FSE 2026.

## Configuration

### 1. Requirements

1. Python version == 3.10.0. Although our method is not strictly dependent on the Python version, the behavior of the `AST` module can vary slightly across different versions. To reproduce our experiments and avoid unexpected behavior, we recommend using Python 3.10.0, which is the same version used in our experimental environment.
2. You should check our dependent python libraries in `requirements.txt` and run `pip install -r requirements.txt` to install them.

### 2. Setting Up with Dataset

Run the following command to extract the dataset.

```bash
tar -xzvf dataset.tar.gz -C ./data
```

### 3. Configuration

If you wish to use your own Large Language Model (LLM) to reproduce our experiments, please modify the corresponding configurations in `src/type_llm/utils/config.py`, including:

1. BASE_URL
2. MODEL
3. API_KEY

If you do not wish to use your own LLM, you can also use our intermediate results (the conversation logs with the LLM) to reproduce the experiments on the example project:

```bash
tar -xzvf LLM_Results.tar.gz -C ./data/intermediate
```

### 4. PYTHONPATH

Run the following command to navigate to the source code directory and set the PYTHONPATH.

```bash
cd src; export PYTHONPATH=.
```

## Run the Example Project (Pre-Commit Hooks)

*If you are using your own LLM, you can edit the value of `projects` in `src/type_llm/utils/config.py`  and follow the same procedure to reproduce our experiments on other projects.

### 1. Generate the Initial EDG

First, we invoke PyAnalyzer to analyze the mapping relationships between variable references and definitions.

```bash
python type_llm/preprocessing/call_PyAnalyzer.py
```

Then, based on these mappings, we construct the initial EDG.

```bash
python type_llm/methods/full_LARRY/PA2EG.py
```

### 2. Start Iteration

```bash
python type_llm/methods/full_LARRY/Entity_Graph.py
```

The partially annotated repositories `pre_commit_hooks_1~9` and the fully annotated repository `pre_commit_hooks` generated in each iteration are saved in the `data/intermediate/validation` directory.

## Evaluation of Annotated Repositories

### 1. Type Accuracy

We use the metrics from TypyBench to evaluate the accuracy of the generated type annotations.

First, build the benchmark dataset.

```bash
python type_llm/evaluation/build_evaluation.py
```

Then, we use the code from TypyBench to calculate the accuracy of the type annotations.

```bash
python -m type_llm.evaluation.typybench.evaluation -n pre_commit_hooks -d ../data/evaluation/projects -p ../data/evaluation/results/PyTIR
```

Finally, we consolidate the accuracy data into the `data/evaluation/EvalResults` directory.

```bash
python type_llm/evaluation/merge_csv.py
```

### 2. # Introduced Type Errors

We use MyPy to check for type errors in the repository after adding the type annotations.

```bash
python type_llm/evaluation/mypy_check.py
```

## Other Statistical Data

Other relevant statistical data can be found in the `static_data` directory.
