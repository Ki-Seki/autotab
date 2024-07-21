# Auto Tabular Completion with In-Context Learning

## Introduction

Welcome to Auto Tabular Completion, a Python script designed to automatically fill in missing values in tabular data using advanced in-context learning techniques. This tool leverages the capabilities of LLMs to predict output values based on given input data.

![Demo](./assets/demo.png)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Ki-Seki/autotab.git
   cd autotab
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Running the Script
    ```bash
    python main.py \
        --file <path_to_your_file>.xlsx \
        --api_key <YOUR_OPENAI_API_KEY> \
        [--base_url <BASE_URL>] \
        [--model <MODEL_NAME>]
    ```

## More

### Optional Arguments

- `--base_url`: Specify the base URL for the OpenAI API. This is useful if you're using a different API endpoint, such as one provided by [Silicon Flow](https://cloud.siliconflow.cn/). Default is the official OpenAI API endpoint.
- `--model`: Choose the model to use for predictions. By default, the script uses `Qwen/Qwen2-7B-Instruct`.

### Input Data Format

Your Excel file should contain columns prefixed with `[Input]` for input variables and `[Output]` for known or expected output values. Any columns not following this naming convention will be ignored during the prediction process.

### Output

After running the script, a new Excel file will be generated in the same directory as the input file, named `file_output.xlsx`, with all missing output values filled in based on the predictions made by the model.
