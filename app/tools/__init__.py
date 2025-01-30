import json
import datetime
import requests
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Any, Optional

import wikipedia
from duckduckgo_search import DDGS

from .everything import search_files as everything_search_files


@dataclass(frozen=True)
class LLMTool:
    name: str
    description: str
    args: list[dict[str, str]]
    function: Callable[..., Any]

    def __repr__(self) -> str:
        return f"{self.name} - {self.description}"

    def get_schema(self) -> dict[str, Any]:
        return json.dumps(
            {
                "name": self.name,
                "description": self.description,
                "args": [arg for arg in self.args],
            },
            indent=2,
            ensure_ascii=False,
        )


def get_current_date() -> str:
    """Returns the current date in the format YYYY-MM-DD."""
    return datetime.date.today().isoformat()


def fetch_webpage(url: str) -> str:
    """Fetch the content of a webpage
    :param url: URL of the webpage to fetch
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return (
            f"Failed to fetch webpage from '{url}', status code: {response.status_code}"
        )


def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a given query and return topic IDs."""
    return ";".join(wikipedia.search(query))


def get_wikipedia_summary(query: str) -> str:
    """Given a topic ID, return details about the Wikipedia page."""
    return wikipedia.page(title=query, auto_suggest=False).summary


def duckduckgo_search(query: str, max_results: int = 10) -> list[dict[str, str]]:
    """Search DuckDuckGo for a given query and return results."""
    return DDGS().text(query, max_results=max_results)


def list_files_and_folders(target_path: str, absolute: bool = False) -> list[str]:
    """List user's files and folders in a given path.
    This tool is useful when you need to interact with user's filesystem, it allows you to know what OS is it, what's username, where certain folders are and etc.

    For Windows, you can get current username by just searching what's inside of 'C:/Users' folder.
    For Linux and macOS, you can get current username by searching what's inside of '/home' folder.


    :param target_path: Target path to list
    :param absolute: If True, returns full paths to files and folders, otherwise relative to the 'target_path' parameter.

    :return: List of file and folder names preseent in the 'target_path'
    """
    return (
        os.listdir(target_path)
        if not absolute
        else [os.path.join(target_path, f) for f in os.listdir(target_path)]
    )


def read_file_contents(target_path: str) -> str:
    """Read file contents

    :param target_path: Target file path to read from

    :return: File contents as a string
    """
    with open(target_path, "r", encoding="utf-8") as file:
        return file.read()


def read_file_contents_bulk(target_paths: list[str]) -> list[str]:
    """Read file contents for multiple files
    :param target_paths: List of target file paths to read from
    :return: List of file contents as strings
    """
    return [read_file_contents(path) for path in target_paths]


def write_file_contents(target_path: str, contents: str) -> str:
    """Write content to a file at the specified path
    :param target_path: Target file path to write content to
    :param contents: Content to write to the file
    """
    with open(target_path, "w", encoding="utf-8") as file:
        file.write(contents)
    return f"Written content to '{target_path}'"


def create_file(target_path: str, contents: Optional[str] = None) -> str:
    """Create a new file at the specified path
    :param target_path: Target file path to create
    :param contents: Content to write to the file if any, empty by default
    """
    with open(target_path, "w", encoding="utf-8") as file:
        file.write(contents or "")

    return f"Created file '{target_path}'"


def create_file_bulk(target_paths: list[str]) -> list[str]:
    """Create multiple files at the specified paths
    :param target_paths: List of target file paths to create
    """
    results: list[str] = []
    for path in target_paths:
        result = create_file(path)
        results.append(result)
    return results


def delete_file(target_path: str) -> str:
    """Delete a file at the specified path
    :param target_path: Target file path to delete
    """
    os.unlink(target_path)
    return f"Deleted file '{target_path}'"


def delete_files_bulk(target_paths: list[str]) -> list[str]:
    """Delete multiple files at the specified paths
    :param target_paths: List of target file paths to delete
    """
    results: list[str] = []
    for path in target_paths:
        result = delete_file(path)
        results.append(result)
    return results


def check_file_existence(target_path: str) -> bool:
    """Check if a file exists at the specified path
    :param target_path: Target file path to check
    """
    return os.path.exists(target_path)


def check_file_existence_bulk(target_paths: list[str]) -> list[bool]:
    """Check if multiple files exist at the specified paths
    :param target_paths: List of target file paths to check
    """
    return {path: check_file_existence(path) for path in target_paths}


def create_folder(target_path: str) -> str:
    """Create a new folder at the specified path
    :param target_path: Target folder path to create
    """
    os.mkdir(target_path)
    return f"Created folder '{target_path}'"


def create_folder_bulk(target_paths: list[str]) -> list[str]:
    """Create multiple folders at the specified paths
    :param target_paths: List of target folder paths to create
    """
    results: list[str] = []
    for path in target_paths:
        result = create_folder(path)
        results.append(result)
    return results


def get_os_username() -> str:
    """Returns the current username of the operating system."""
    return os.getlogin()


def execute_shell(command: str) -> str:
    """Execute a shell command and return the output

    :param command: Shell command to execute

    :return: stdout output of the shell command
    """
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        return result.stdout.strip() or result.stderr.strip() or "OK"
    except Exception as e:
        return str(e)


def execute_shell_bulk(commands: list[str]) -> dict[str, str]:
    """Execute a shell command and return the output. Preffered way for executing multiple shell commands.

    :param commands: List of shell commands to execute

    :return: Dictionary of stdout/stderr outputs of the shell commands
    """
    results: dict[str, str] = {}
    for command in commands:
        try:
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            results[command] = result.stdout.strip() or result.stderr.strip()
        except Exception as e:
            results[command] = str(e)
    return results


def download_file(url: str, target_path: str) -> str:
    """Downloads file from the provided URL to a target path on user's machine

    :param url: target file URL to download from
    :param target_path: destination file path to download to

    :return: Download result"""
    try:
        response = requests.get(url)
        with open(target_path, "wb") as file:
            file.write(response.content)
        return f"Downloaded {url} successfully"
    except Exception as e:
        return f"Error while downloading {url}: {e}"


def download_file_bulk(urls: list[str], target_paths: list[str]) -> dict[str, str]:
    """Downloads files from the provided URLs to a target paths on user's machine

    :param urls: target file URLs to download from
    :param target_path: destination file paths to download to

    :return: Download result for each url"""
    result: dict[str, str] = {}
    for idx, url in enumerate(urls):
        result[url] = download_file(url, target_paths[idx])
    return result


def search_files(query: str) -> list[dict[str, str]]:
    """Searches for files via \"Everything\" application using the provided query and returns a list of dictionaries containing the file path, size, and modified date.
    This is a preferred way to search for files quickly and efficiently.

    :param query: The query to search for files.
    :return: A list of dictionaries containing the file path, size, and modified date."""
    return everything_search_files(query)
