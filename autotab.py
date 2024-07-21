import argparse
import concurrent.futures
import os
import re

import openai
import pandas as pd


def load_excel(file_path):
    """Load the Excel file and identify input and output columns."""
    df = pd.read_excel(file_path)
    input_columns = [col for col in df.columns if col.startswith("[Input]")]
    output_columns = [col for col in df.columns if col.startswith("[Output]")]
    return df, input_columns, output_columns


def derive_in_context(data, input_columns, output_columns):
    """Derive in-context learning data from the given DataFrame."""
    n = len(data.dropna(subset=output_columns))
    in_context = ""
    for i in range(n):
        in_context += "".join(
            f"<{col.replace('[Input] ', '')}>{data[col].iloc[i]}</{col.replace('[Input] ', '')}>\n"
            for col in input_columns
        ) + "".join(
            f"<{col.replace('[Output] ', '')}>{data[col].iloc[i]}</{col.replace('[Output] ', '')}>\n"
            for col in output_columns
        )
    return in_context, n


def predict_output(in_context, input_data, input_columns, api_key, model, base_url):
    """Predict the output values for the given input data using the OpenAI API."""
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    query = in_context + "".join(
        f"<{col.replace('[Input] ', '')}>{input_data[col]}</{col.replace('[Input] ', '')}>\n"
        for col in input_columns
    )
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": query}], max_tokens=64, n=1
    )
    predictions = response.choices[0].message.content.strip()
    return predictions


def extract_fields(prediction, output_columns):
    """Extract fields from the prediction text based on output columns."""
    extracted = {}
    for col in output_columns:
        tag = col.replace("[Output] ", "")
        match = re.search(f"<{tag}>(.*?)</{tag}>", prediction)
        extracted[tag] = match.group(1) if match else ""
    return extracted


def process_file(file_path, api_key, model, base_url):
    """Process the entire Excel file and predict output values for missing entries."""
    df, input_columns, output_columns = load_excel(file_path)
    in_context, n = derive_in_context(df, input_columns, output_columns)

    results = [None] * (len(df) - n)  # Pre-allocate a list for results
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                predict_output,
                in_context,
                df.iloc[i],
                input_columns,
                api_key,
                model,
                base_url,
            ): i
            - n
            for i in range(n, len(df))
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                prediction = future.result()
                extracted_fields = extract_fields(prediction, output_columns)
                results[idx] = extracted_fields
            except Exception as e:
                results[idx] = {
                    col.replace("[Output] ", ""): f"Error: {e}"
                    for col in output_columns
                }

    # Assign results to the DataFrame
    for i, extracted_fields in enumerate(results, start=n):
        for col in output_columns:
            tag = col.replace("[Output] ", "")
            df.at[i, col] = extracted_fields.get(tag, "")

    # Save the updated DataFrame to a new Excel file
    output_file_path = os.path.splitext(file_path)[0] + "_output.xlsx"
    df.to_excel(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")


def main():
    """Main function to parse arguments and execute the processing."""
    parser = argparse.ArgumentParser(
        description="Auto Tabular Completion with In-Context Learning"
    )
    parser.add_argument("--file", required=True, help="Path to the Excel file")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument(
        "--base_url", required=False, help="OpenAI API base URL", default=None
    )
    parser.add_argument(
        "--model",
        required=False,
        help="OpenAI model to use",
        default="Qwen/Qwen2-7B-Instruct",
    )

    args = parser.parse_args(
        "--file data/sample_nlp.xlsx --api_key sk-exhahhjfqyanmwewndukcqtrpegfdbwszkjucvcpajdufiah --base_url https://api.siliconflow.cn/v1".split()
    )

    process_file(args.file, args.api_key, args.model, args.base_url)


if __name__ == "__main__":
    main()
