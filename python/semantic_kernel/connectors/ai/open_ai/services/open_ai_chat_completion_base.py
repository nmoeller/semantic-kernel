# Copyright (c) Microsoft. All rights reserved.

import logging
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice

from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionCallChoiceConfiguration
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_handler import OpenAIHandler
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.streaming_text_content import StreamingTextContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.contents.utils.finish_reason import FinishReason
from semantic_kernel.exceptions import (
    ServiceInvalidResponseError,
)
from semantic_kernel.functions.kernel_function_metadata import KernelFunctionMetadata

logger: logging.Logger = logging.getLogger(__name__)


class InvokeTermination(Exception):
    """Exception for termination of function invocation."""

    pass


class OpenAIChatCompletionBase(OpenAIHandler, ChatCompletionClientBase):
    """OpenAI Chat completion class."""

    # region Overriding base class methods
    # most of the methods are overridden from the ChatCompletionClientBase class, otherwise it is mentioned

    # override from AIServiceClientBase
    def get_prompt_execution_settings_class(self) -> "PromptExecutionSettings":
        """Create a request settings object."""
        return OpenAIChatPromptExecutionSettings
    
    async def get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: OpenAIChatPromptExecutionSettings,
        **kwargs: Any,
    ) -> list["ChatMessageContent"]:
        """Executes a chat completion request and returns the result.

        Args:
            chat_history (ChatHistory): The chat history to use for the chat completion.
            settings (OpenAIChatPromptExecutionSettings | AzureChatPromptExecutionSettings): The settings to use
                for the chat completion request.
            kwargs (Dict[str, Any]): The optional arguments.

        Returns:
            List[ChatMessageContent]: The completion result(s).
        """
        if not hasattr(settings, "ai_model_id"):
            settings.ai_model_id = self.ai_model_id

        settings.messages = self._prepare_chat_history_for_request(chat_history)
        response = await self._send_request(request_settings=settings)
        response_metadata = self._get_metadata_from_chat_response(response)
        return [self._create_chat_message_content(response, choice, response_metadata) for choice in response.choices]
    
    async def get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: OpenAIChatPromptExecutionSettings,
        **kwargs: Any,
    ) -> AsyncGenerator[list[StreamingChatMessageContent | None], Any]:
        """Executes a streaming chat completion request and returns the result.

        Args:
            chat_history (ChatHistory): The chat history to use for the chat completion.
            settings (OpenAIChatPromptExecutionSettings | AzureChatPromptExecutionSettings): The settings to use
                for the chat completion request.
            kwargs (Dict[str, Any]): The optional arguments.

        Yields:
            List[StreamingChatMessageContent]: A stream of
                StreamingChatMessageContent when using Azure.
        """
        if not hasattr(settings, "ai_model_id"):
            settings.ai_model_id = self.ai_model_id

        settings.stream = True
        settings.messages = self._prepare_chat_history_for_request(chat_history)
        response = await self._send_request(request_settings=settings)
        if not isinstance(response, AsyncStream):
            raise ServiceInvalidResponseError("Expected an AsyncStream[ChatCompletionChunk] response.")
        async for chunk in response:
            if len(chunk.choices) == 0:
                continue
            chunk_metadata = self._get_metadata_from_streaming_chat_response(chunk)
            yield [
                self._create_streaming_chat_message_content(chunk, choice, chunk_metadata) for choice in chunk.choices
            ]

    # endregion

    # region content creation

    def _create_chat_message_content(
        self, response: ChatCompletion, choice: Choice, response_metadata: dict[str, Any]
    ) -> "ChatMessageContent":
        """Create a chat message content object from a choice."""
        metadata = self._get_metadata_from_chat_choice(choice)
        metadata.update(response_metadata)

        items: list[Any] = self._get_tool_calls_from_chat_choice(choice)
        items.extend(self._get_function_call_from_chat_choice(choice))
        if choice.message.content:
            items.append(TextContent(text=choice.message.content))

        return ChatMessageContent(
            inner_content=response,
            ai_model_id=self.ai_model_id,
            metadata=metadata,
            role=AuthorRole(choice.message.role),
            items=items,
            finish_reason=(FinishReason(choice.finish_reason) if choice.finish_reason else None),
        )

    def _create_streaming_chat_message_content(
        self,
        chunk: ChatCompletionChunk,
        choice: ChunkChoice,
        chunk_metadata: dict[str, Any],
    ) -> StreamingChatMessageContent | None:
        """Create a streaming chat message content object from a choice."""
        metadata = self._get_metadata_from_chat_choice(choice)
        metadata.update(chunk_metadata)

        items: list[Any] = self._get_tool_calls_from_chat_choice(choice)
        items.extend(self._get_function_call_from_chat_choice(choice))
        if choice.delta.content is not None:
            items.append(StreamingTextContent(choice_index=choice.index, text=choice.delta.content))
        return StreamingChatMessageContent(
            choice_index=choice.index,
            inner_content=chunk,
            ai_model_id=self.ai_model_id,
            metadata=metadata,
            role=(AuthorRole(choice.delta.role) if choice.delta.role else AuthorRole.ASSISTANT),
            finish_reason=(FinishReason(choice.finish_reason) if choice.finish_reason else None),
            items=items,
        )

    def _get_metadata_from_chat_response(self, response: ChatCompletion) -> dict[str, Any]:
        """Get metadata from a chat response."""
        return {
            "id": response.id,
            "created": response.created,
            "system_fingerprint": response.system_fingerprint,
            "usage": getattr(response, "usage", None),
        }

    def _get_metadata_from_streaming_chat_response(self, response: ChatCompletionChunk) -> dict[str, Any]:
        """Get metadata from a streaming chat response."""
        return {
            "id": response.id,
            "created": response.created,
            "system_fingerprint": response.system_fingerprint,
        }

    def _get_metadata_from_chat_choice(self, choice: Choice | ChunkChoice) -> dict[str, Any]:
        """Get metadata from a chat choice."""
        return {
            "logprobs": getattr(choice, "logprobs", None),
        }

    def _get_tool_calls_from_chat_choice(self, choice: Choice | ChunkChoice) -> list[FunctionCallContent]:
        """Get tool calls from a chat choice."""
        content = choice.message if isinstance(choice, Choice) else choice.delta
        if content.tool_calls is None:
            return []
        return [
            FunctionCallContent(
                id=tool.id,
                index=getattr(tool, "index", None),
                name=tool.function.name,
                arguments=tool.function.arguments,
            )
            for tool in content.tool_calls
        ]

    def _get_function_call_from_chat_choice(self, choice: Choice | ChunkChoice) -> list[FunctionCallContent]:
        """Get a function call from a chat choice."""
        content = choice.message if isinstance(choice, Choice) else choice.delta
        if content.function_call is None:
            return []
        return [
            FunctionCallContent(
                id="legacy_function_call", name=content.function_call.name, arguments=content.function_call.arguments
            )
        ]

    # endregion
    # region function calling conifg

    def update_settings_from_function_call_configuration(
        self,
        function_choice_configuration: "FunctionCallChoiceConfiguration",
        settings: "OpenAIChatPromptExecutionSettings",
        type: str,
    ) -> None:
        """Update the settings from a FunctionChoiceConfiguration."""
        if function_choice_configuration.available_functions:
            settings.tool_choice = type
            settings.tools = [
                self.kernel_function_metadata_to_function_call_format(f)
                for f in function_choice_configuration.available_functions
            ]

    def kernel_function_metadata_to_function_call_format(
        self,
        metadata: KernelFunctionMetadata,
    ) -> dict[str, Any]:
        """Convert the kernel function metadata to function calling format."""
        return {
            "type": "function",
            "function": {
                "name": metadata.fully_qualified_name,
                "description": metadata.description or "",
                "parameters": {
                    "type": "object",
                    "properties": {param.name: param.schema_data for param in metadata.parameters if param.is_required},
                    "required": [p.name for p in metadata.parameters if p.is_required],
                },
            },
        }
