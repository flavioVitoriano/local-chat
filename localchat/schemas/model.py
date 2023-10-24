from pydantic import BaseModel, field_validator
import os
from typing import List, Optional


class ModelArgs(BaseModel):
    temperature: float
    top_p: float
    max_tokens: int
    n_gpu_layers: Optional[int]
    n_batch: Optional[int]


class ChatModel(BaseModel):
    chat_model_path: str
    persist_directory_path: str
    documents_directory_path: str
    prompt_template: str
    input_variables: List[str] = ["question", "documents"]
    llama_args: ModelArgs

    @field_validator("chat_model_path")
    @classmethod
    def validate_valid_path(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError("File does not exists")

        if not os.path.isfile(v):
            raise ValueError("Path is not a valid file")

        return v
