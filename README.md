### Description

- An attempt to use deep learning for the [Two Sigma Financial Modeling Challenge](https://www.kaggle.com/c/two-sigma-financial-modeling).
- **The results do not look promising: a validation R score of -0.0024 after 4 epochs**.

### Usage

1. Download and install [neon](https://github.com/NervanaSystems/neon) **1.8.1**

    ```
    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    git checkout v1.8.1
    make
    source .venv/bin/activate
    ```
2. Verify neon installation

    Make sure that this command does not result in any errors:
    ```
    ./examples/mnist_mlp.py -e1
    ```

3. Install prerequisites

    ```
    pip install pandas sklearn
    ```
4. Download the data files from [Kaggle](https://www.kaggle.com/c/two-sigma-financial-modeling/data):

    Save all files to a directory (referred to as /path/to/data below) and unzip the .zip file.

5. Clone this repository

    ```
    git clone https://github.com/anlthms/fmc-2017.git
    cd fmc-2017
    ```
6. Train a model and validate

    ```
    ./run.py -w </path/to/data> -e 4 -r 0 -quick
    ```

### Notes

- Omit `-quick` to use the entire data for training and validation.
