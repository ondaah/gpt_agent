import re
import json
from pydantic import BaseModel
from typing import Any, Optional, Self
from abc import ABC, abstractmethod

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

    def is_valid(self) -> bool:
        return (self.thoughts and self.tool) or (self.thoughts and self.response)


class BaseConversationProtocol(ABC):
    SCHEMA_EXAMPLE = ""

    @classmethod
    @abstractmethod
    def parse(self, message: str) -> StructuredMessage:
        pass

    @classmethod
    @abstractmethod
    def serialize(self, message: StructuredMessage, new_line: bool = False) -> str:
        pass


class XMLConversationProtocol(BaseConversationProtocol):
    SCHEMA_EXAMPLE = """<thoughts>Your reasoning should happen here</thoughts>
<tool>{{tool_name}} or leave empty if no tool is needed</tool>
<tool_args>{{["input required for the tool"]}} or empty list if no input is needed</tool_args>
<response>null if using a tool, otherwise your final response</response>"""

    @classmethod
    def parse(cls, text: str) -> StructuredMessage:
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

        return StructuredMessage(
            thoughts=thoughts, tool=tool, tool_args=tool_args, response=response
        )

    @classmethod
    def serialize(cls, message: StructuredMessage, new_line: bool = False) -> str:
        return "<thoughts>{}</thoughts>{nl}<tool>{}</tool>{nl}<tool_args>{}</tool_args>{nl}<response>{}</response>".format(
            message.thoughts,
            message.tool.lower(),
            json.dumps(message.tool_args),
            message.response,
            nl="\n" if new_line else "",
        )


class JSONConversationProtocol(BaseConversationProtocol):
    SCHEMA_EXAMPLE = """{
  "thoughts": "Your reasoning should happen here",
  "tool": "{{tool_name}} or leave empty if no tool is needed",
  "tool_args": {{["input required for the tool"]}} or empty list if no input is needed,
  "response": "null if using a tool, otherwise your final response"
}"""

    @classmethod
    def parse(cls, text: str) -> StructuredMessage:
        data: dict[str, Any] = json.loads(text)
        return StructuredMessage(
            thoughts=data.get("thoughts"),
            tool=data.get("tool"),
            tool_args=data.get("tool_args"),
            response=data.get("response"),
        )

    @classmethod
    def serialize(cls, message: StructuredMessage, new_line: bool = False) -> str:
        return json.dumps(
            {
                "thoughts": message.thoughts,
                "tool": message.tool,
                "tool_args": message.tool_args,
                "response": message.response,
            },
            indent=2 if new_line else None,
            ensure_ascii=False,
        )
