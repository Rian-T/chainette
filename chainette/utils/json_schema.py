from typing import Type
from pydantic import BaseModel
import json

__all__ = ["generate_json_output_prompt"]

def generate_json_output_prompt(output_model: Type[BaseModel]) -> str:
    """
    Generates a string to append to the system prompt, instructing the LLM
    to output JSON data according to the provided Pydantic model's schema.
    """
    schema = output_model.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    return (
        "Your final response MUST be a JSON object that conforms to the following JSON schema.\n"
        #"Focus on providing the data as specified by the schema.\n"
        #"Do NOT output the schema definition itself or any other explanatory text outside the JSON object.\n\n"
        "JSON Schema to follow:\n"
        "```json\n"
        f"{schema_str}\n"
        "```\n"
        #"Ensure your entire response is a single, valid JSON object that adheres to this schema."
    ) 