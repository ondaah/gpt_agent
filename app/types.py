import re
import json
from pydantic import BaseModel
from typing import Any, Optional, Self

_REGEXP_FLAGS = re.IGNORECASE | re.MULTILINE | re.DOTALL

THOUGHTS_EXPR = re.compile(r"<thoughts>(.*?)</thoughts>", _REGEXP_FLAGS)
TOOL_EXPR = re.compile(r"<tool>(.*?)</tool>", _REGEXP_FLAGS)
ARGUMENTS_EXPR = re.compile(r"<tool_args>(.*?)</tool_args>", _REGEXP_FLAGS)
RESPONSE_EXPR = re.compile(r"<response>(.*?)</response>", _REGEXP_FLAGS)


class StructuredMessage(BaseModel):
    thoughts: Optional[str]
    tool: Optional[str]
    tool_args: list[Any]
    response: Optional[str]

    @classmethod
    def parse_text(cls, text: str) -> Self:
        if match := THOUGHTS_EXPR.search(text):
            thoughts = match.group(1).strip()
        else:
            thoughts = None

        if match := TOOL_EXPR.search(text):
            tool = match.group(1).strip()
        else:
            tool = None

        if match := ARGUMENTS_EXPR.search(text):
            try:
                tool_args = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                tool_args = []
        else:
            tool_args = []

        if match := RESPONSE_EXPR.search(text):
            response = match.group(1).strip()
        else:
            response = None

        if not response:
            response = thoughts

        return cls(thoughts=thoughts, tool=tool, tool_args=tool_args, response=response)

    def serialize(self) -> str:
        return "<thoughts>{}</thoughts><tool>{}</tool><tool_args>{}</tool_args><response>{}</response>".format(
            self.thoughts,
            self.tool.lower(),
            json.dumps(self.tool_args),
            self.response,
        )

    def is_valid(self) -> bool:
        return (self.thoughts and self.tool) or (self.thoughts and self.response)
