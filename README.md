# Auto Tabular Completion with In-Context Learning

![](https://img.shields.io/github/last-commit/Ki-Seki/autotab?color=green) [![](https://img.shields.io/badge/-%F0%9F%A4%97%20HuggingFace%20Space-orange?style=flat)](https://huggingface.co/spaces/Ki-Seki/AutoTab) [![](https://img.shields.io/badge/-%F0%9F%A4%96%20ModelScope%20Space-blue?style=flat)](https://www.modelscope.cn/studios/KiSeki/AutoTab) 

Welcome to Auto Tabular Completion. It automatically fill in missing values in tabular data using in-context learning techniques. This tool leverages the capabilities of LLMs to predict output values based on given input data.

![Demo](./assets/demo.png)

## Usage

Check [demo.ipynb](demo.ipynb) for more details. 

Check [![](https://img.shields.io/badge/-%F0%9F%A4%97%20HuggingFace%20Space-orange?style=flat)](https://huggingface.co/spaces/Ki-Seki/AutoTab) and [![](https://img.shields.io/badge/-%F0%9F%A4%96%20ModelScope%20Space-blue?style=flat)](https://www.modelscope.cn/studios/KiSeki/AutoTab) for direct usage.

> [!Note]
> **Give it a try with my personal key in the two demo notebooks (valid until 2024-11-01).** [Silicon Flow](https://cloud.siliconflow.cn/) provides free API. You can sign up and get your own API key.

## Input Format

Your Excel file should contain columns prefixed with `[Input] ` for input variables and `[Output] ` for known or expected output values. Any columns not following this naming convention will be ignored during the prediction process.

## TODOs

- **Use asyncio and ClientSession**: Refactor the code to use `asyncio` and `ClientSession` for asynchronous processing instead of using `ThreadPoolExecutor`. This will improve efficiency and scalability for handling API requests and data processing.

- **Save Data to SQLite Table**: Update the code to save results into a SQLite database instead of an Excel file. This will involve creating a SQLite table and implementing the necessary functions to insert data into it.
