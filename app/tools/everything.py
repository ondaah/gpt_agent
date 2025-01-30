import re
import requests
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import asyncio
import aiohttp
from bs4 import BeautifulSoup, ResultSet, Tag


def parse_size(size_str: str) -> int:
    """Преобразует строку размера в байты."""
    size_str = size_str.upper().replace(" ", "")
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    if match := re.match(r"(\d+(?:\.\d+)?)([KMGT]?B)", size_str):
        value, unit = match.groups()
        return int(float(value) * units[unit])
    return 0


@dataclass(frozen=True)
class SearchResult:
    path: Path
    is_folder: bool
    size: int
    modified: datetime

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "is_folder": self.is_folder,
            "size": self.size,
            "modified": self.modified.isoformat(),
        }


async def fetch_page_async(
    session: aiohttp.ClientSession, query: str, offset: int
) -> tuple[int, str]:
    """Асинхронно запрашивает страницу поиска Everything."""
    url = f"http://localhost:5432/?search={query}&offset={offset}"
    async with session.get(url, headers={"Accept-Encoding": "gzip"}) as response:
        return offset, await response.text()


def fetch_page_sync(query: str, offset: int) -> tuple[int, str]:
    """Синхронно запрашивает страницу поиска Everything."""
    url = f"http://localhost:5432/?search={query}&offset={offset}"
    response = requests.get(url, headers={"Accept-Encoding": "gzip"})
    return offset, response.text


def parse_page(html: str) -> tuple[list[SearchResult], int]:
    """Извлекает файлы, их тип и дату изменения из HTML."""
    parser = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []

    for tr in parser.select("tr[class^=trdata]"):
        tds: ResultSet[Tag] = tr.find_all("td")
        type_class = tds[0].get("class", [])
        is_folder = "folder" in type_class
        filename = tds[0].text
        path = Path(tds[1].text, filename)
        size = parse_size(tds[2].text.strip())
        modified = datetime.strptime(tds[3].text.strip(), "%m/%d/%Y %I:%M %p")
        results.append(SearchResult(path, is_folder, size, modified))

    max_offset = 0
    if spans := parser.find_all("span", {"class": "nav"}):
        if a := spans[-1].find("a"):
            if match := re.search(r"offset=(\d+)", a.get("href")):
                max_offset = int(match.group(1))

    return results, max_offset


async def search_files_async(query: str) -> list[SearchResult]:
    """Асинхронно ищет файлы во всех страницах Everything."""
    files_by_offset: dict[int, list[SearchResult]] = {}
    seen_offsets: set[int] = set()
    tasks: dict[int, asyncio.Task[tuple[int, str]]] = {}

    async with aiohttp.ClientSession() as session:
        tasks[0] = asyncio.create_task(fetch_page_async(session, query, 0))

        while tasks:
            done, _ = await asyncio.wait(
                tasks.values(), return_when=asyncio.FIRST_COMPLETED
            )

            for future in done:
                offset, html = await future
                tasks.pop(offset)

                if offset in seen_offsets:
                    continue

                seen_offsets.add(offset)

                page_files, max_offset = parse_page(html)
                files_by_offset[offset] = page_files

                for new_offset in range(offset + 32, max_offset + 1, 32):
                    if new_offset not in seen_offsets and new_offset not in tasks:
                        tasks[new_offset] = asyncio.create_task(
                            fetch_page_async(session, query, new_offset)
                        )

    return [
        file for offset in sorted(files_by_offset) for file in files_by_offset[offset]
    ]


def search_files(query: str) -> list[SearchResult]:
    """Синхронно ищет файлы во всех страницах Everything."""
    files_by_offset: dict[int, list[SearchResult]] = {}
    seen_offsets: set[int] = set()
    offsets_to_fetch = [0]

    while offsets_to_fetch:
        offset = offsets_to_fetch.pop(0)
        if offset in seen_offsets:
            continue

        seen_offsets.add(offset)
        _, html = fetch_page_sync(query, offset)
        page_files, max_offset = parse_page(html)
        files_by_offset[offset] = page_files

        for new_offset in range(offset + 32, max_offset + 1, 32):
            if new_offset not in seen_offsets:
                offsets_to_fetch.append(new_offset)

    return [
        file for offset in sorted(files_by_offset) for file in files_by_offset[offset]
    ]
