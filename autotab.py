import re

import openai
import pandas as pd
from tqdm import tqdm


class AutoTab:
    def __init__(
        self,
        in_file_path: str,
        out_file_path: str,
        max_examples: int,
        model_name: str,
        api_key: str,
        base_url: str,
        generation_config: dict,
        save_every: int,
        instruction: str,
    ):
        self.in_file_path = in_file_path
        self.out_file_path = out_file_path
        self.max_examples = max_examples
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.generation_config = generation_config
        self.save_every = save_every
        self.instruction = instruction

    # ─── IO ───────────────────────────────────────────────────────────────

    def load_excel(self) -> tuple[pd.DataFrame, list, list]:
        """Load the Excel file and identify input and output fields."""
        df = pd.read_excel(self.in_file_path)
        input_fields = [col for col in df.columns if col.startswith("[Input] ")]
        output_fields = [col for col in df.columns if col.startswith("[Output] ")]
        return df, input_fields, output_fields

    # ─── LLM ──────────────────────────────────────────────────────────────

    def openai_request(self, query: str) -> str:
        """Make a request to an OpenAI-format API."""
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": query}],
            **self.generation_config,
        )
        str_response = response.choices[0].message.content.strip()
        return str_response

    # ─── In-Context Learning ──────────────────────────────────────────────

    def derive_incontext(
        self, data: pd.DataFrame, input_columns: list[str], output_columns: list[str]
    ) -> str:
        """Derive the in-context prompt with angle brackets."""
        n = min(self.max_examples, len(data.dropna(subset=output_columns)))
        in_context = ""
        for i in range(n):
            in_context += "".join(
                f"<{col.replace('[Input] ', '')}>{data[col].iloc[i]}</{col.replace('[Input] ', '')}>\n"
                for col in input_columns
            )
            in_context += "".join(
                f"<{col.replace('[Output] ', '')}>{data[col].iloc[i]}</{col.replace('[Output] ', '')}>\n"
                for col in output_columns
            )
            in_context += "\n"
        self.in_context = in_context
        return in_context

    def predict_output(
        self, in_context: str, input_data: pd.DataFrame, input_fields: str
    ):
        """Predict the output values for the given input data using the API."""
        query = (
            self.instruction
            + "\n\n"
            + in_context
            + "".join(
                f"<{col.replace('[Input] ', '')}>{input_data[col]}</{col.replace('[Input] ', '')}>\n"
                for col in input_fields
            )
        )
        self.query_example = query
        output = self.openai_request(query)
        return output

    def extract_fields(
        self, response: str, output_columns: list[str]
    ) -> dict[str, str]:
        """Extract fields from the response text based on output columns."""
        extracted = {}
        for col in output_columns:
            field = col.replace("[Output] ", "")
            match = re.search(f"<{field}>(.*?)</{field}>", response)
            extracted[col] = match.group(1) if match else ""
        return extracted

    # ─── Engine ───────────────────────────────────────────────────────────

    def run(self):
        data, input_fields, output_fields = self.load_excel()
        in_context = self.derive_incontext(data, input_fields, output_fields)

        num_existed_examples = len(data.dropna(subset=output_fields))

        for i in tqdm(range(num_existed_examples, len(data))):
            prediction = self.predict_output(in_context, data.iloc[i], input_fields)
            extracted_fields = self.extract_fields(prediction, output_fields)
            for field_name in output_fields:
                data.at[i, field_name] = extracted_fields.get(field_name, "")
            if i % self.save_every == 0:
                data.to_excel(self.out_file_path, index=False)
        self.data = data
        data.to_excel(self.out_file_path, index=False)
        print(f"Results saved to {self.out_file_path}")
