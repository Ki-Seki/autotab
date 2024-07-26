# Auto Tabular Completion with In-Context Learning

Welcome to Auto Tabular Completion. It automatically fill in missing values in tabular data using in-context learning techniques. This tool leverages the capabilities of LLMs to predict output values based on given input data.



![Demo1](./assets/demo1.png)

## Usage

Check [demo1.ipynb](demo1.ipynb) and [demo2.ipynb](demo2.ipynb) for more details.

> [!Note]
> **Give it a try with my personal key in the two demo notebooks (valid until 2024-11-01).** [Silicon Flow](https://cloud.siliconflow.cn/) provides free API. You can sign up and get your own API key.

## Input Format

Your Excel file should contain columns prefixed with `[Input] ` for input variables and `[Output] ` for known or expected output values. Any columns not following this naming convention will be ignored during the prediction process.
