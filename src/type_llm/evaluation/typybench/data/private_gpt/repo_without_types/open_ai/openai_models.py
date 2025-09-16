import time
import uuid
from collections.abc import Iterator
from typing import Literal

from llama_index.core.llms import ChatResponse, CompletionResponse
from pydantic import BaseModel, Field

from private_gpt.server.chunks.chunks_service import Chunk


class OpenAIDelta(BaseModel):
    """A piece of completion that needs to be concatenated to get the full message."""

    content = None


class OpenAIMessage(BaseModel):
    """Inference result, with the source of the message.

    Role could be the assistant or system
    (providing a default response, not AI generated).
    """

    role = Field(default="user")
    content = None


class OpenAIChoice(BaseModel):
    """Response from AI.

    Either the delta or the message will be present, but never both.
    Sources used will be returned in case context retrieval was enabled.
    """

    finish_reason = Field(examples=["stop"])
    delta = None
    message = None
    sources = None
    index = 0


class OpenAICompletion(BaseModel):
    """Clone of OpenAI Completion model.

    For more information see: https://platform.openai.com/docs/api-reference/chat/object
    """

    id = None
    object = Field(default="completion")
    created = Field(..., examples=[1623340000])
    model = None
    choices = None

    @classmethod
    def from_text(
        cls, text, finish_reason = None, sources = None):
        return OpenAICompletion(
            id=str(uuid.uuid4()),
            object="completion",
            created=int(time.time()),
            model="private-gpt",
            choices=[
                OpenAIChoice(
                    message=OpenAIMessage(role="assistant", content=text),
                    finish_reason=finish_reason,
                    sources=sources,
                )
            ],
        )

    @classmethod
    def json_from_delta(
        cls, *,
        text, finish_reason = None, sources = None):
        chunk = OpenAICompletion(
            id=str(uuid.uuid4()),
            object="completion.chunk",
            created=int(time.time()),
            model="private-gpt",
            choices=[
                OpenAIChoice(
                    delta=OpenAIDelta(content=text),
                    finish_reason=finish_reason,
                    sources=sources,
                )
            ],
        )

        return chunk.model_dump_json()


def to_openai_response(
    response, sources = None):
    if isinstance(response, ChatResponse):
        return OpenAICompletion.from_text(response.delta, finish_reason="stop")
    else:
        return OpenAICompletion.from_text(
            response, finish_reason="stop", sources=sources
        )


def to_openai_sse_stream(
    response_generator, sources = None):
    for response in response_generator:
        if isinstance(response, CompletionResponse | ChatResponse):
            yield f"data: {OpenAICompletion.json_from_delta(text=response.delta)}\n\n"
        else:
            yield f"data: {OpenAICompletion.json_from_delta(text=response, sources=sources)}\n\n"
    yield f"data: {OpenAICompletion.json_from_delta(text='', finish_reason='stop')}\n\n"
    yield "data: [DONE]\n\n"
