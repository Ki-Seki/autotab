import gradio as gr

from autotab import AutoTab
import json
import time
import pandas as pd


def convert_seconds_to_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def auto_tabulator_completion(
    in_file,
    instruction,
    max_examples,
    model_name,
    generation_config,
    save_every,
    api_key,
    base_url,
) -> tuple[str, str, str, pd.DataFrame]:
    output_file_name = "ouput.xlsx"
    autotab = AutoTab(
        in_file_path=in_file.name,
        instruction=instruction,
        out_file_path=output_file_name,
        max_examples=max_examples,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        generation_config=json.loads(generation_config),
        save_every=save_every,
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
