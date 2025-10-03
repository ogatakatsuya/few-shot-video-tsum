from abc import ABCMeta, abstractmethod
from pathlib import Path


class Prompt:
    def __init__(self, video_path: Path, text: str):
        self.video_path = video_path
        self.text = text


class BaseTsumGenerator(metaclass=ABCMeta):
    @abstractmethod
    def generate(self, prompts: list[Prompt]) -> str:
        raise NotImplementedError("Subclasses must implement the generate method.")
