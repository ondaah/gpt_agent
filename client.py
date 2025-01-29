import re
import json
import requests
from typing import Callable, Any, Optional
from inspect import signature

from rich.console import Console

from app.types import StructuredMessage
from app.tools import (
    LLMTool,
    get_current_date,
    fetch_webpage,
    search_wikipedia,
    get_wikipedia_summary,
    duckduckgo_search,
    list_files_and_folders,
    read_file_contents,
    read_file_contents_bulk,
    write_file_contents,
    create_file,
    create_file_bulk,
    delete_file,
    delete_files_bulk,
)

console = Console()


class LLM:
    def __init__(
        self,
        model_name: str,
        tools: list[Callable[..., Any]] = None,
        model_endpoint: str = "http://127.0.0.1:1337/v1/chat/completions",
    ):
        self.model_name = model_name
        self.model_endpoint = model_endpoint

        self.tools: list[LLMTool] = []
        for tool in tools:
            self.register_tool(tool)

        self.history: list[dict[str, str]] = []

    def generate_system_prompt(self) -> str:
        tool_list: list[str] = []

        for tool in self.tools:
            tool_list.append(tool.get_schema())

        return """YOU MUST RESPOND ONLY IN THIS STRICT FORMAT:
<thoughts>Your reasoning should happen here</thoughts>
<tool>{{tool_name}} or leave empty if no tool is needed</tool>
<tool_args>{{["input required for the tool"]}} or empty list if no input is needed</tool_args>
<response>Your answer here</response>

DO NOT include any explanations. Also, you must respond and think in user's native language.

You have access to the following tools:
{tool_list}

For example, if you want to get today's date, you must use the action "get_current_date" without including the tool name in your response text. For example:
<thoughts>I need to know the current date to answer the user's question.</thoughts>
<tool>get_current_date</tool>
<tool_args>[]</tool_args>
<response></response>

For the second example, if you want to find some information on Wikipedia, you can use the action "wikipedia_search" with the desired query. For example:
<thoughts>I need to find information on the topic 'Python'.</thoughts>
<tool>wikipedia_search</tool>
<tool_args>["Python"]</tool_args>
<response></response>

If you need to provide a collection of arguments (i.e. create multiple files) use the following 'action_input' syntax:
<thoughts>I need to create multiple files in the user's file system.</thoughts>
<tool>create_file_bulk</tool>
<tool_args>[["C:\\doc1.tt", "C:\\doc2.txt"]]</tool_args>
<response></response>

This instructions apply to any other tool so use them accordingly.
""".format(tool_list="\n".join(tool_list))

    def chat(self, prompt: str, role: str = "user") -> StructuredMessage:
        self.history.append({"role": role, "content": prompt})

        response = self.send_request()
        # console.print("[green bold]Ответ[/green bold]")
        # print(response)
        # console.print("[green bold]/Ответ[/green bold]")

        response_text = self.clean_response(response)
        response = StructuredMessage.parse_text(response_text)

        # console.print("[yellow bold]Ответ[/yellow bold]")
        # print(response)
        # console.print("[yellow bold]/Ответ[/yellow bold]")

        self.history.append({"role": "assistant", "content": response_text})

        if not response.is_valid():
            raise ValueError(f"Invalid response: {response_text}")

        if response.thoughts:
            console.print(f"[dim white]{response.thoughts}[/dim white]")

        if response.tool:
            console.print(
                "[dim white]Использую инструмент {}({})[/dim white]".format(
                    response.tool, response.tool_args
                )
            )

            if tool := self.get_tool_by_name(response.tool):
                try:
                    tool_result = tool.function(*response.tool_args)
                except Exception as e:
                    return self.chat(
                        f'<thoughts>"{response.tool}" tool returned error: "{e}", proceeding further...</thoughts>',
                        role="assistant",
                    )
                return self.chat(
                    f'"<thoughts>{response.tool}" tool returned result: "{tool_result}", proceeding further...</thoughts>',
                    role="assistant",
                )
            else:
                print(f"Tool '{response.tool}' not found.")

        return response

    def send_request(self):
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": self.generate_system_prompt()}]
            + self.history,
        }

        response = requests.post(self.model_endpoint, json=payload)
        return response.json()

    @staticmethod
    def clean_response(response: dict):
        raw_response = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        # raw_response = re.sub(
        #     r"```json(.+?)```", "\\1", raw_response, flags=re.M | re.DOTALL
        # )
        match = re.search(r"({.+})", raw_response, flags=re.M | re.DOTALL)
        if match:
            raw_response = match.group(1).strip()
        return raw_response

    def register_tool(self, func: Callable) -> LLMTool:
        func_args: list[dict[str, Any]] = []

        annotations_mapping: dict[Any, str] = {
            str: "string",
            int: "integer",
            float: "float",
            bool: "boolean",
            list: "list",
            dict: "dict",
            Callable: "function",
            Any: "unknown",
        }

        # Получаем информацию о аргументах метода
        sig = signature(func)
        for param in sig.parameters.values():
            func_args.append(
                {
                    "name": param.name,
                    "type": annotations_mapping.get(param.annotation, "Unknown")
                    if param.annotation is not sig.empty
                    else "Unknown",
                    "has_default": param.default is not sig.empty,
                    "default": param.default
                    if param.default is not sig.empty
                    else None,
                }
            )

        tool = LLMTool(
            name=func.__name__,
            description=func.__doc__,
            args=func_args,
            function=func,
        )
        self.tools.append(tool)
        return tool

    def get_tool_by_name(self, name: str) -> Optional[LLMTool]:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


def main():
    llm = LLM(
        model_name="gpt-4o-mini",
        model_endpoint="http://127.0.0.1:1337/v1/chat/completions",
        # model_name="llama3.1:8b",
        # model_endpoint="http://127.0.0.1:11434/v1/chat/completions",
        tools=[
            get_current_date,
            # web
            fetch_webpage,
            search_wikipedia,
            get_wikipedia_summary,
            duckduckgo_search,
            # fs
            list_files_and_folders,
            read_file_contents,
            read_file_contents_bulk,
            write_file_contents,
            create_file,
            create_file_bulk,
            delete_file,
            delete_files_bulk,
        ],
    )

    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(llm.generate_system_prompt())

    try:
        while True:
            question = input("> ")
            response = llm.chat(question)
            console.print(f"[green]{response.response}[/green]")
    except KeyboardInterrupt:
        pass
    finally:
        with open("history.json", "w", encoding="utf-8") as file:
            json.dump(
                [{"role": "system", "content": llm.generate_system_prompt()}]
                + llm.history,
                file,
                indent=2,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
