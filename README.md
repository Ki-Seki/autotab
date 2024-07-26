# Auto Tabular Completion with In-Context Learning

![](https://img.shields.io/github/last-commit/Ki-Seki/autotab?color=green) [![](https://img.shields.io/badge/-%F0%9F%A4%97%20HuggingFace%20Space-orange?style=flat)](https://huggingface.co/spaces/Ki-Seki/AutoTab)

Welcome to Auto Tabular Completion. It automatically fill in missing values in tabular data using in-context learning techniques. This tool leverages the capabilities of LLMs to predict output values based on given input data.

![Demo1](./assets/demo1.png)

## Usage

Check [![](https://img.shields.io/badge/-%F0%9F%A4%97%20HuggingFace%20Space-orange?style=flat)](https://huggingface.co/spaces/Ki-Seki/AutoTab) for direct usage.

Check [demo1.ipynb](demo1.ipynb) and [demo2.ipynb](demo2.ipynb) for more details.

> [!Note]
> **Give it a try with my personal key in the two demo notebooks (valid until 2024-11-01).** [Silicon Flow](https://cloud.siliconflow.cn/) provides free API. You can sign up and get your own API key.

## Input Format

Your Excel file should contain columns prefixed with `[Input] ` for input variables and `[Output] ` for known or expected output values. Any columns not following this naming convention will be ignored during the prediction process.
