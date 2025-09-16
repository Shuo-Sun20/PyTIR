from openai import OpenAI, APIError
from openai import RateLimitError, InternalServerError
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
    ChatCompletionAssistantMessageParam,
)
from typing import Callable, List, Optional
from copy import deepcopy
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_log,
    after_log,
)
from pathlib import Path
from type_llm.utils import silent_JDump
import json
import concurrent.futures
import time
import logging
from type_llm.utils.config import BASE_URL, API_KEY, MODEL
from type_llm.utils.log_manager import setup_logger
from type_llm.utils import is_valid_msg, json2stub

MAX_CONCURRENT = 12
RETRY_TIME = 3
logger = setup_logger(with_console=False)


client = OpenAI(base_url= BASE_URL, api_key= API_KEY[0])

def _purge_conversation(conversation: list[ChatCompletionMessageParam]):
    """
    Purge the reasoning content from the conversation
    """
    messages = deepcopy(conversation)
    for message in messages:
        for key in list(message.keys()):
            if key not in ["content", "role"]:
                del message[key]
    return messages

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((RateLimitError, InternalServerError, APIError)),
    before=before_log(logger, logging.WARNING),
    after=after_log(logger, logging.WARNING),
    reraise=True,
)
def _query_llm(messages: list[ChatCompletionMessageParam]):
    conversation = _purge_conversation(messages)
    logger.info("Preparing to query LLM")
    logger.debug(f"Messages: {conversation}")

    response = client.chat.completions.create(
        model=MODEL,
        messages=conversation,
        stream=True,
        # max_tokens = MAX_TOKENS,
        temperature = 0.1,
        timeout = 1200
    )

    collected_chunks: list[ChatCompletionChunk] = []
    collected_content: list[str | None] = []
    collected_reasoning_content: list[str | None] = []
    normal_finished = False
    for chunk in response:
        # logger.debug(f"Received chunk: {chunk}")
        collected_chunks.append(chunk)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        collected_content.append(delta.content)
        if "reasoning_content" in delta.model_extra:
            collected_reasoning_content.append(delta.model_extra["reasoning_content"])
        finish_reason = chunk.choices[0].finish_reason
        if finish_reason == "stop":
            normal_finished = True
        elif finish_reason == "length":
            logger.warning("LLM query finished due to length limit")
    if not normal_finished:
        return None
    full_content = "".join([m for m in collected_content if m is not None])
    full_reasoning_content = "".join(
        [m for m in collected_reasoning_content if m is not None]
    )
    logger.info("LLM query completed")
    logger.debug(f"Full content: {full_content}")
    logger.debug(f"Full reasoning content: {full_reasoning_content}")
    
    conversation.append(
        ChatCompletionAssistantMessageParam(
            content=full_content,
            role="assistant",
            reasoning_content=full_reasoning_content,
        )
    )

    return conversation


class UnExpectedResponseException(BaseException):
    pass

def _query_llm_with_retry(
    messages: list[ChatCompletionMessageParam],
    retry_times: int,
    validate_func: Callable[[str], Optional[str]],
    file_name=""
):
    logger.info(f"Query LLM with retry {retry_times} times-------{file_name}")
    while True:
        current_messages = messages.copy()
        for times in range(retry_times):
            logger.info(f"Retry {times + 1} / {retry_times}-----{messages}------{file_name}")
            try:
                conversation = _query_llm(current_messages)
            except:
                continue
            logger.debug(f"RET Conversation:{conversation}------{file_name}")
            if conversation is None:
                return None
            content = conversation[-1]["content"]
            logger.debug(f'content:{content}-------{file_name}')
            msg = validate_func(content)
            if not msg:
                logger.info(f"Get valid response from LLM-------{file_name}")
                return conversation
            else:
                logger.error(f"Get unexpected response from LLM:{conversation}-------{msg}------{file_name}")
                try:
                    if len(current_messages) == len(messages):
                        current_messages.append(conversation[-1])
                        current_messages.append({"role":"user", "content":f"I met an error when trying to extract needed information from your answer: {msg}. Please regenerate the answer."})
                    else:
                        current_messages[-2] = conversation[-1]
                        current_messages[-1] = {"role":"user", "content":f"I met an error when trying to extract needed information from your answer: {msg}. Please regenerate the answer."}
                except Exception as e:
                    logger.error(f"Error when appending message: {msg} : {e}")
                    raise e
    