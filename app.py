import gradio as gr

from autotab import AutoTab
import json


def auto_tabulator_completion(
    in_file,
    instruction,
    max_examples,
    model_name,
    generation_config,
    save_every,
):
    output_file_name = "ouput.xlsx"
    autotab = AutoTab(
        in_file_path=in_file.name,
        instruction=instruction,
        out_file_path=output_file_name,
        max_examples=max_examples,
        model_name=model_name,
        api_key="sk-exhahhjfqyanmwewndukcqtrpegfdbwszkjucvcpajdufiah",
        base_url="https://public-beta-api.siliconflow.cn/v1",
        generation_config=json.loads(generation_config),
        save_every=save_every,
    )
    autotab.run()
    return output_file_name, autotab.data[:15]


# Gradio interface
inputs = [
    gr.File(label="Input Excel File"),
    gr.Textbox(
        value="You are a helpful assistant. Help me finish the task.",
        label="Instruction",
    ),
    gr.Slider(value=5, minimum=1, maximum=100, label="Max Examples"),
    gr.Textbox(value="Qwen/Qwen2-7B-Instruct", label="Model Name"),
    gr.Textbox(
        value='{"temperature": 0, "max_tokens": 128}',
        label="Generation Config in Dict",
    ),
    gr.Slider(value=10, minimum=1, maximum=1000, label="Save Every N Steps"),
]

outputs = [
    gr.File(label="Output Excel File"),
    gr.Dataframe(label="First 15 rows."),
]

gr.Interface(
    fn=auto_tabulator_completion,
    inputs=inputs,
    outputs=outputs,
    title="Auto Tabulator Completion",
    description="Automatically complete missing output values in tabular data based on in-context learning.",
).launch()
