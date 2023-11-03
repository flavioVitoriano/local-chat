from pydantic import BaseModel, field_validator
import os
from typing import List, Optional


class ModelArgs(BaseModel):
    temperature: float = 0.5
    top_p: float = 0.9
    max_tokens: int = 500
    n_gpu_layers: Optional[int] = 0
    n_batch: Optional[int] = 8
    n_ctx: int = 512


class ChatModel(BaseModel):
    chat_model_path: str
    persist_directory_path: str
    documents_directory_path: str
    prompt_template_file: str
    input_variables: List[str] = ["question", "documents"]
    llama_args: ModelArgs

    @field_validator("chat_model_path", "prompt_template_file")
    @classmethod
    def validate_valid_path(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError("File does not exists")

        if not os.path.isfile(v):
            raise ValueError("Path is not a valid file")

        return v

    @property
    def prompt_template(self):
        with open(self.prompt_template_file, "r") as f:
            return f.read()
