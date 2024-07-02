# Copyright (c) Microsoft. All rights reserved.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI

from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletionBase
from semantic_kernel.contents import AuthorRole, ChatMessageContent, StreamingChatMessageContent, TextContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.kernel import Kernel


async def mock_async_process_chat_stream_response(arg1, response, tool_call_behavior, chat_history, kernel, arguments):
    mock_content = MagicMock(spec=StreamingChatMessageContent)
    yield [mock_content], None


@pytest.mark.asyncio
async def test_complete_chat_stream(kernel: Kernel):
    chat_history = MagicMock()
    settings = MagicMock()
    settings.number_of_responses = 1
    mock_response = MagicMock()
    arguments = KernelArguments()

    with (
        patch(
            "semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion_base.OpenAIChatCompletionBase.get_streaming_chat_message_contents",
            return_value=mock_response,
        ) as mock_send_chat_stream_request,
    ):
        chat_completion_base = OpenAIChatCompletionBase(
            ai_model_id="test_model_id", service_id="test", client=MagicMock(spec=AsyncOpenAI)
        )

        async for content in chat_completion_base.invoke_streaming(
            chat_history, settings, kernel=kernel, arguments=arguments
        ):
            assert content is not None

        mock_send_chat_stream_request.assert_called_with(chat_history, settings, kernel=kernel, arguments=arguments)


@pytest.mark.parametrize("tool_call", [False, True])
@pytest.mark.asyncio
async def test_complete_chat_function_call_behavior(tool_call, kernel: Kernel):
    chat_history = MagicMock(spec=ChatHistory)
    chat_history.messages = []
    settings = MagicMock(spec=OpenAIChatPromptExecutionSettings)
    settings.number_of_responses = 1
    settings.function_call_behavior = None
    settings.function_choice_behavior = None
    mock_function_call = MagicMock(spec=FunctionCallContent)
    mock_text = MagicMock(spec=TextContent)
    mock_message = ChatMessageContent(
        role=AuthorRole.ASSISTANT, items=[mock_function_call] if tool_call else [mock_text]
    )
    mock_message_content = [mock_message]
    arguments = KernelArguments()

    if tool_call:
        settings.function_call_behavior = MagicMock(spec=FunctionCallBehavior.AutoInvokeKernelFunctions())
        settings.function_call_behavior.auto_invoke_kernel_functions = True
        settings.function_call_behavior.max_auto_invoke_attempts = 5
        chat_history.messages = [mock_message]

    with (
        patch(
            "semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion_base.OpenAIChatCompletionBase.get_chat_message_contents",
            return_value=mock_message_content,
        ),
        patch(
            "semantic_kernel.kernel.Kernel.invoke_function_call",
            new_callable=AsyncMock,
        ) as mock_process_function_call,
    ):
        chat_completion_base = OpenAIChatCompletionBase(
            ai_model_id="test_model_id", service_id="test", client=MagicMock(spec=AsyncOpenAI)
        )

        result = await chat_completion_base.invoke(
            chat_history, settings, kernel=kernel, arguments=arguments
        )

        assert result is not None

        if tool_call:
            mock_process_function_call.assert_awaited()
        else:
            mock_process_function_call.assert_not_awaited()


@pytest.mark.parametrize("tool_call", [False, True])
@pytest.mark.asyncio
async def test_complete_chat_function_choice_behavior(tool_call, kernel: Kernel):
    chat_history = MagicMock(spec=ChatHistory)
    chat_history.messages = []
    settings = MagicMock(spec=OpenAIChatPromptExecutionSettings)
    settings.number_of_responses = 1
    settings.function_choice_behavior = None
    mock_function_call = MagicMock(spec=FunctionCallContent)
    mock_text = MagicMock(spec=TextContent)
    mock_message = ChatMessageContent(
        role=AuthorRole.ASSISTANT, items=[mock_function_call] if tool_call else [mock_text]
    )
    mock_message_content = [mock_message]
    arguments = KernelArguments()

    if tool_call:
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        chat_history.messages = [mock_message]

    with (
        patch(
            "semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion_base.OpenAIChatCompletionBase.get_chat_message_contents",
            return_value=mock_message_content,
        ) as mock_get_chat_message_contents,
        patch(
            "semantic_kernel.kernel.Kernel.invoke_function_call",
            new_callable=AsyncMock,
        ) as mock_process_function_call,
    ):
        chat_completion_base = OpenAIChatCompletionBase(
            ai_model_id="test_model_id", service_id="test", client=MagicMock(spec=AsyncOpenAI)
        )

        result = await chat_completion_base.invoke(
            chat_history, settings, kernel=kernel, arguments=arguments
        )

        assert result is not None
        mock_get_chat_message_contents.assert_called_with(chat_history, settings, kernel=kernel, arguments=arguments)

        if tool_call:
            mock_process_function_call.assert_awaited()
        else:
            mock_process_function_call.assert_not_awaited()
