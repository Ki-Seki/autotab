import json
import time

import gradio as gr
import pandas as pd

from autotab import AutoTab


def auto_tabulator_completion(
    in_file_path: str,
    instruction: str,
    max_examples: int,
    model_name: str,
    generation_config: dict,
    request_interval: float,
    save_every: int,
    api_key: str,
    base_url: str,
) -> tuple[str, str, str, pd.DataFrame]:
    output_file_name = "ouput.xlsx"
    autotab = AutoTab(
        in_file_path=in_file_path,
        out_file_path=output_file_name,
        instruction=instruction,
        max_examples=max_examples,
        model_name=model_name,
        generation_config=json.loads(generation_config),
        request_interval=request_interval,
        save_every=save_every,
        api_key=api_key,
        base_url=base_url,
    )
    start = time.time()
    autotab.run()
    time_taken = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))

    return time_taken, output_file_name, autotab.query_example, autotab.data[:15]


# Gradio interface
inputs = [
    gr.File(label="Input Excel File"),
    gr.Textbox(
        value="You are a helpful assistant. Help me finish the task.",
        label="Instruction",
    ),
    gr.Slider(value=5, minimum=1, maximum=50, step=1, label="Max Examples"),
    gr.Textbox(value="Qwen/Qwen2-7B-Instruct", label="Model Name"),
    gr.Textbox(
        value='{"temperature": 0, "max_tokens": 128}',
        label="Generation Config in Dict",
    ),
    gr.Slider(value=0.1, minimum=0, maximum=10, label="Request Interval in Seconds"),
    gr.Slider(value=100, minimum=1, maximum=1000, step=1, label="Save Every N Steps"),
    gr.Textbox(
        value="sk-exhahhjfqyanmwewndukcqtrpegfdbwszkjucvcpajdufiah", label="API Key"
    ),
    gr.Textbox(value="https://public-beta-api.siliconflow.cn/v1", label="Base URL"),
]

outputs = [
    gr.Textbox(label="Time Taken"),
    gr.File(label="Output Excel File"),
    gr.Textbox(label="Query Example"),
    gr.Dataframe(label="First 15 rows."),
]

gr.Interface(
    fn=auto_tabulator_completion,
    inputs=inputs,
    outputs=outputs,
    title="Auto Tabulator Completion",
    description="Automatically complete missing output values in tabular data based on in-context learning.",
).launch()
