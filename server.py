import asyncio
from asyncio import WindowsSelectorEventLoopPolicy
from fastapi import FastAPI
from pydantic import BaseModel
from g4f.client import AsyncClient
from g4f.Provider import PollinationsAI, OIVSCode, DDG, RetryProvider


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    temperature: float = 0


app = FastAPI()

asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
client = AsyncClient(provider=RetryProvider([PollinationsAI, OIVSCode, DDG]))


@app.post("/v1/chat/completions")
async def create_chat_completion(request: CompletionRequest):
    print(request)
    response = await client.chat.completions.create(
        messages=request.messages, model=request.model
    )
    return {
        "id": response.id,
        "object": response.object,
        "created": response.created,
        # "model": "gpt-4o-mini",
        "model": None,
        "provider": None,
        "choices": [
            {
                "index": choice.index,
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content,
                },
                "finish_reason": "length",
            }
            for choice in response.choices
        ],
        "usage": {
            "prompt_tokens": response.usage["prompt_tokens"],
            "completion_tokens": response.usage["completion_tokens"],
            "total_tokens": response.usage["total_tokens"],
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=1337)
