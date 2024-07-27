import re
import time
from concurrent.futures import ThreadPoolExecutor

import openai
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm


class AutoTab:
    def __init__(
        self,
        in_file_path: str,
        out_file_path: str,
        instruction: str,
        max_examples: int,
        model_name: str,
        generation_config: dict,
        request_interval: float,
        save_every: int,
        api_key: str,
        base_url: str,
    ):
        self.in_file_path = in_file_path
        self.out_file_path = out_file_path
        self.instruction = instruction
        self.max_examples = max_examples
        self.model_name = model_name
        self.generation_config = generation_config
        self.request_interval = request_interval
        self.save_every = save_every
        self.api_key = api_key
        self.base_url = base_url

    # ─── IO ───────────────────────────────────────────────────────────────

    def load_excel(self) -> tuple[pd.DataFrame, list, list]:
        """Load the Excel file and identify input and output fields."""
        df = pd.read_excel(self.in_file_path)
        input_fields = [col for col in df.columns if col.startswith("[Input] ")]
        output_fields = [col for col in df.columns if col.startswith("[Output] ")]
        return df, input_fields, output_fields

    # ─── LLM ──────────────────────────────────────────────────────────────

    @retry(wait=wait_random_exponential(min=20, max=60), stop=stop_after_attempt(6))
    def openai_request(self, query: str) -> str:
        """Make a request to an OpenAI-format API."""
        time.sleep(self.request_interval)
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

    def _predict_and_extract(self, i: int) -> dict[str, str]:
        """Helper function to predict and extract fields for a single row."""
        prediction = self.predict_output(
            self.in_context, self.data.iloc[i], self.input_fields
        )
        extracted_fields = self.extract_fields(prediction, self.output_fields)
        return extracted_fields

    def batch_prediction(self, start_index: int, end_index: int):
        """Process a batch of predictions asynchronously."""
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._predict_and_extract, range(start_index, end_index))
            )
        for i, extracted_fields in zip(range(start_index, end_index), results):
            for field_name in self.output_fields:
                self.data.at[i, field_name] = extracted_fields.get(field_name, "")

    def run(self):
        self.data, self.input_fields, self.output_fields = self.load_excel()
        self.in_context = self.derive_incontext(
            self.data, self.input_fields, self.output_fields
        )

        self.num_data = len(self.data)
        self.num_examples = len(self.data.dropna(subset=self.output_fields))

        tqdm_bar = tqdm(total=self.num_data - self.num_examples, leave=False)
        for start in range(self.num_examples, self.num_data, self.save_every):
            tqdm_bar.update(min(self.save_every, self.num_data - start))
            end = min(start + self.save_every, self.num_data)
            try:
                self.batch_prediction(start, end)
            except Exception as e:
                print(e)
            self.data.to_excel(self.out_file_path, index=False)
        self.data.to_excel(self.out_file_path, index=False)
        print(f"Results saved to {self.out_file_path}")
