from pathlib import Path
from typing import Union

from google.genai import types


class GeminiTextPrompt:
    def __init__(self, message: str):
        self.message = message

    def to_part(self) -> types.Part:
        return types.Part(text=self.message)


class GeminiYoutubePrompt:
    def __init__(self, video_path: Path):
        self.video_path = video_path

    def to_part(self) -> types.Part:
        return types.Part(
            file_data=types.FileData(
                file_uri=f"https://www.youtube.com/watch?v={self._extract_video_id(self.video_path)}"
            )
        )

    def _extract_video_id(self, path: Path) -> str:
        """
        Extract the video ID from the YouTube video URL.

        example:
        args: path = Path("data/youtube_video_dumps/v__abcdefghijk.mp4")
        return: "abcdefghijk"
        """
        video_id = path.name.split("v_")[-1].split(".")[0]
        if video_id.startswith("_"):
            video_id = video_id[1:]
        return video_id


class GeminiPrompt:
    def __init__(
        self,
        messages: list[Union[GeminiYoutubePrompt, GeminiTextPrompt]],
    ):
        self.messages = messages

    def to_parts(self) -> list[types.Part]:
        return [message.to_part() for message in self.messages]
