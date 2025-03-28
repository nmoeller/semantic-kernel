﻿// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Amazon.BedrockAgent;
using Amazon.BedrockAgentRuntime;
using Amazon.BedrockAgentRuntime.Model;
using Microsoft.SemanticKernel.Agents.Extensions;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Diagnostics;

namespace Microsoft.SemanticKernel.Agents.Bedrock;

/// <summary>
/// Provides a specialized <see cref="KernelAgent"/> for the Bedrock Agent service.
/// </summary>
public class BedrockAgent : KernelAgent
{
    /// <summary>
    /// The client used to interact with the Bedrock Agent service.
    /// </summary>
    public AmazonBedrockAgentClient Client { get; }

    /// <summary>
    /// The client used to interact with the Bedrock Agent runtime service.
    /// </summary>
    public AmazonBedrockAgentRuntimeClient RuntimeClient { get; }

    internal readonly Amazon.BedrockAgent.Model.Agent AgentModel;

    /// <summary>
    /// There is a default alias created by Bedrock for the working draft version of the agent.
    /// https://docs.aws.amazon.com/bedrock/latest/userguide/agents-deploy.html
    /// </summary>
    public static readonly string WorkingDraftAgentAlias = "TSTALIASID";

    /// <summary>
    /// Initializes a new instance of the <see cref="BedrockAgent"/> class.
    /// Unlike other types of agents in Semantic Kernel, prompt templates are not supported for Bedrock agents,
    /// since Bedrock agents don't support using an alternative instruction in runtime.
    /// </summary>
    /// <param name="agentModel">The agent model of an agent that exists on the Bedrock Agent service.</param>
    /// <param name="client">A client used to interact with the Bedrock Agent service.</param>
    /// <param name="runtimeClient">A client used to interact with the Bedrock Agent runtime service.</param>
    public BedrockAgent(
        Amazon.BedrockAgent.Model.Agent agentModel,
        AmazonBedrockAgentClient client,
        AmazonBedrockAgentRuntimeClient runtimeClient)
    {
        this.AgentModel = agentModel;
        this.Client = client;
        this.RuntimeClient = runtimeClient;

        this.Id = agentModel.AgentId;
        this.Name = agentModel.AgentName;
        this.Description = agentModel.Description;
        this.Instructions = agentModel.Instruction;
    }

    #region static methods

    /// <summary>
    /// Convenient method to create an unique session id.
    /// </summary>
    public static string CreateSessionId()
    {
        return Guid.NewGuid().ToString();
    }

    #endregion

    #region public methods

    /// <inheritdoc/>
    public override IAsyncEnumerable<AgentResponseItem<ChatMessageContent>> InvokeAsync(
        ICollection<ChatMessageContent> messages,
        AgentThread? thread = null,
        AgentInvokeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        return this.InvokeAsync(messages, thread, options, WorkingDraftAgentAlias, cancellationToken);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<AgentResponseItem<ChatMessageContent>> InvokeAsync(
        ICollection<ChatMessageContent> messages,
        AgentThread? thread = null,
        AgentInvokeOptions? options = null,
        string? agentAliasId = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Verify.NotNull(messages, nameof(messages));
        if (messages.Count == 0)
        {
            throw new InvalidOperationException("The Bedrock agent requires a message to be invoked.");
        }

        // Create a thread if needed
        var bedrockThread = await this.EnsureThreadExistsWithMessagesAsync(
            messages,
            thread,
            () => new BedrockAgentThread(this.RuntimeClient),
            cancellationToken).ConfigureAwait(false);

        // Ensure that the last message provided is a user message
        string? message = this.ExtractUserMessage(messages.Last());

        // Build session state with conversation history if needed
        SessionState sessionState = this.ExtractSessionState(messages);

        var invokeAgentRequest = new InvokeAgentRequest
        {
            AgentAliasId = agentAliasId ?? WorkingDraftAgentAlias,
            SessionState = sessionState,
            AgentId = this.Id,
            SessionId = bedrockThread.Id,
            InputText = message,
        };

        var arguments = this.MergeArguments(options?.KernelArguments);
        var invokeResults = this.InvokeAsync(invokeAgentRequest, arguments, cancellationToken);

        // Return the results to the caller in AgentResponseItems.
        await foreach (var result in invokeResults.ConfigureAwait(false))
        {
            await this.NotifyThreadOfNewMessage(bedrockThread, result, cancellationToken).ConfigureAwait(false);
            yield return new(result, bedrockThread);
        }
    }

    /// <inheritdoc/>
    public override async IAsyncEnumerable<AgentResponseItem<StreamingChatMessageContent>> InvokeStreamingAsync(
        ICollection<ChatMessageContent> messages,
        AgentThread? thread = null,
        AgentInvokeOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Verify.NotNull(messages, nameof(messages));
        if (messages.Count == 0)
        {
            throw new InvalidOperationException("The Bedrock agent requires a message to be invoked.");
        }

        // Create a thread if needed
        var bedrockThread = await this.EnsureThreadExistsWithMessagesAsync(
            messages,
            thread,
            () => new BedrockAgentThread(this.RuntimeClient),
            cancellationToken).ConfigureAwait(false);

        // Ensure that the last message provided is a user message
        string? message = this.ExtractUserMessage(messages.Last());

        // Build session state with conversation history if needed
        SessionState sessionState = this.ExtractSessionState(messages);

        var invokeAgentRequest = new InvokeAgentRequest
        {
            AgentAliasId = WorkingDraftAgentAlias,
            SessionState = sessionState,
            AgentId = this.Id,
            SessionId = bedrockThread.Id,
            InputText = message,
        };

        var arguments = this.MergeArguments(options?.KernelArguments);
        var invokeResults = this.InvokeStreamingAsync(invokeAgentRequest, arguments, cancellationToken);

        // Return the results to the caller in AgentResponseItems.
        await foreach (var result in invokeResults.ConfigureAwait(false))
        {
            yield return new(result, bedrockThread);
        }
    }

    /// <summary>
    /// Invoke the Bedrock agent with the given message.
    /// </summary>
    /// <param name="sessionId">The session id.</param>
    /// <param name="message">The message to send to the agent.</param>
    /// <param name="arguments">The arguments to use when invoking the agent.</param>
    /// <param name="agentAliasId">The alias id of the agent to use. The default is the working draft alias id.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> to monitor for cancellation requests. The default is <see cref="CancellationToken.None"/>.</param>
    /// <returns>An <see cref="IAsyncEnumerable{T}"/> of <see cref="ChatMessageContent"/>.</returns>
    public IAsyncEnumerable<ChatMessageContent> InvokeAsync(
        string sessionId,
        string message,
        KernelArguments? arguments,
        string? agentAliasId = null,
        CancellationToken cancellationToken = default)
    {
        var invokeAgentRequest = new InvokeAgentRequest
        {
            AgentAliasId = agentAliasId ?? WorkingDraftAgentAlias,
            AgentId = this.Id,
            SessionId = sessionId,
            InputText = message,
        };

        return this.InvokeAsync(invokeAgentRequest, arguments, cancellationToken);
    }

    /// <summary>
    /// Invoke the Bedrock agent with the given request. Use this method when you want to customize the request.
    /// </summary>
    /// <param name="invokeAgentRequest">The request to send to the agent.</param>
    /// <param name="arguments">The arguments to use when invoking the agent.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> to monitor for cancellation requests. The default is <see cref="CancellationToken.None"/>.</param>
    public IAsyncEnumerable<ChatMessageContent> InvokeAsync(
        InvokeAgentRequest invokeAgentRequest,
        KernelArguments? arguments,
        CancellationToken cancellationToken = default)
    {
        return invokeAgentRequest.StreamingConfigurations != null && (invokeAgentRequest.StreamingConfigurations.StreamFinalResponse ?? false)
            ? throw new ArgumentException("The streaming configuration must be null for non-streaming responses.")
            : ActivityExtensions.RunWithActivityAsync(
                () => ModelDiagnostics.StartAgentInvocationActivity(this.Id, this.GetDisplayName(), this.Description),
                InvokeInternal,
                cancellationToken);

        // Collect all responses from the agent and return them as a single chat message content since this
        // is a non-streaming API.
        // The Bedrock Agent API streams beck different types of responses, i.e. text, files, metadata, etc.
        // The Bedrock Agent API also won't stream back any content when it needs to call a function. It will
        // only start streaming back content after the function has been called and the response is ready.
        async IAsyncEnumerable<ChatMessageContent> InvokeInternal()
        {
            ChatMessageContentItemCollection items = [];
            string content = "";
            Dictionary<string, object?> metadata = [];
            List<object?> innerContents = [];

            await foreach (var message in this.InternalInvokeAsync(invokeAgentRequest, arguments, cancellationToken).ConfigureAwait(false))
            {
                items.AddRange(message.Items);
                content += message.Content ?? "";
                if (message.Metadata != null)
                {
                    foreach (var key in message.Metadata.Keys)
                    {
                        metadata[key] = message.Metadata[key];
                    }
                }
                innerContents.Add(message.InnerContent);
            }

            if (content.Length == 0)
            {
                throw new KernelException("No content was returned from the agent.");
            }

            var chatMessageContent = new ChatMessageContent(AuthorRole.Assistant, content)
            {
                AuthorName = this.GetDisplayName(),
                Items = items,
                ModelId = this.AgentModel.FoundationModel,
                Metadata = metadata,
                InnerContent = innerContents,
            };

            yield return chatMessageContent;
        }
    }

    /// <summary>
    /// Invoke the Bedrock agent with the given request and streaming response.
    /// </summary>
    /// <param name="sessionId">The session id.</param>
    /// <param name="message">The message to send to the agent.</param>
    /// <param name="arguments">The arguments to use when invoking the agent.</param>
    /// <param name="agentAliasId">The alias id of the agent to use. The default is the working draft alias id.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> to monitor for cancellation requests. The default is <see cref="CancellationToken.None"/>.</param>
    /// <returns>An <see cref="IAsyncEnumerable{T}"/> of <see cref="ChatMessageContent"/>.</returns>
    public IAsyncEnumerable<StreamingChatMessageContent> InvokeStreamingAsync(
        string sessionId,
        string message,
        KernelArguments? arguments,
        string? agentAliasId = null,
        CancellationToken cancellationToken = default)
    {
        var invokeAgentRequest = new InvokeAgentRequest
        {
            AgentAliasId = agentAliasId ?? WorkingDraftAgentAlias,
            AgentId = this.Id,
            SessionId = sessionId,
            InputText = message,
            StreamingConfigurations = new()
            {
                StreamFinalResponse = true,
            },
        };

        return this.InvokeStreamingAsync(invokeAgentRequest, arguments, cancellationToken);
    }

    /// <summary>
    /// Invoke the Bedrock agent with the given request and streaming response. Use this method when you want to customize the request.
    /// </summary>
    /// <param name="invokeAgentRequest">The request to send to the agent.</param>
    /// <param name="arguments">The arguments to use when invoking the agent.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> to monitor for cancellation requests. The default is <see cref="CancellationToken.None"/>.</param>
    /// <returns>An <see cref="IAsyncEnumerable{T}"/> of <see cref="StreamingChatMessageContent"/>.</returns>
    public IAsyncEnumerable<StreamingChatMessageContent> InvokeStreamingAsync(
        InvokeAgentRequest invokeAgentRequest,
        KernelArguments? arguments,
        CancellationToken cancellationToken = default)
    {
        if (invokeAgentRequest.StreamingConfigurations == null)
        {
            invokeAgentRequest.StreamingConfigurations = new()
            {
                StreamFinalResponse = true,
            };
        }
        else if (!(invokeAgentRequest.StreamingConfigurations.StreamFinalResponse ?? false))
        {
            throw new ArgumentException("The streaming configuration must have StreamFinalResponse set to true.");
        }

        return ActivityExtensions.RunWithActivityAsync(
            () => ModelDiagnostics.StartAgentInvocationActivity(this.Id, this.GetDisplayName(), this.Description),
            InvokeInternal,
            cancellationToken);

        async IAsyncEnumerable<StreamingChatMessageContent> InvokeInternal()
        {
            // The Bedrock agent service has the same API for both streaming and non-streaming responses.
            // We are invoking the same method as the non-streaming response with the streaming configuration set,
            // and converting the chat message content to streaming chat message content.
            await foreach (var chatMessageContent in this.InternalInvokeAsync(invokeAgentRequest, arguments, cancellationToken).ConfigureAwait(false))
            {
                yield return new StreamingChatMessageContent(chatMessageContent.Role, chatMessageContent.Content)
                {
                    AuthorName = chatMessageContent.AuthorName,
                    ModelId = chatMessageContent.ModelId,
                    InnerContent = chatMessageContent.InnerContent,
                    Metadata = chatMessageContent.Metadata,
                };
            }
        }
    }

    #endregion

    /// <inheritdoc/>
    protected override IEnumerable<string> GetChannelKeys()
    {
        // Return the channel keys for the BedrockAgent
        yield return typeof(BedrockAgentChannel).FullName!;
    }

    /// <inheritdoc/>
    protected override Task<AgentChannel> CreateChannelAsync(CancellationToken cancellationToken)
    {
        // Create and return a new BedrockAgentChannel
        return Task.FromResult<AgentChannel>(new BedrockAgentChannel());
    }

    /// <inheritdoc/>
    protected override Task<AgentChannel> RestoreChannelAsync(string channelState, CancellationToken cancellationToken)
    {
        // Restore and return a BedrockAgentChannel from the given state
        return Task.FromResult<AgentChannel>(new BedrockAgentChannel());
    }

    #region internal methods

    internal string CodeInterpreterActionGroupSignature { get => $"{this.GetDisplayName()}_CodeInterpreter"; }
    internal string KernelFunctionActionGroupSignature { get => $"{this.GetDisplayName()}_KernelFunctions"; }
    internal string UseInputActionGroupSignature { get => $"{this.GetDisplayName()}_UserInput"; }

    #endregion

    #region private methods

    private string ExtractUserMessage(ChatMessageContent chatMessageContent)
    {
        if (!chatMessageContent.Role.Equals(AuthorRole.User))
        {
            throw new InvalidOperationException("Bedrock agents must be invoked with a user message");
        }

        return chatMessageContent.Content ?? "";
    }

    private SessionState ExtractSessionState(ICollection<ChatMessageContent> messages)
    {
        // If there is more than one message provided, add all but the last message to the session state
        SessionState sessionState = new();
        if (messages.Count > 1)
        {
            List<Amazon.BedrockAgentRuntime.Model.Message> messageHistory = [];
            for (int i = 0; i < messages.Count - 1; i++)
            {
                var currentMessage = messages.ElementAt(i);
                messageHistory.Add(this.ToBedrockMessage(currentMessage));
            }

            sessionState.ConversationHistory = new ConversationHistory() { Messages = messageHistory };
        }

        return sessionState;
    }

    private Amazon.BedrockAgentRuntime.Model.Message ToBedrockMessage(ChatMessageContent chatMessageContent)
    {
        return new Amazon.BedrockAgentRuntime.Model.Message()
        {
            Role = this.MapBedrockAgentUser(chatMessageContent.Role),
            Content = [new() { Text = chatMessageContent.Content }]
        };
    }

    private Amazon.BedrockAgentRuntime.ConversationRole MapBedrockAgentUser(AuthorRole authorRole)
    {
        if (authorRole == AuthorRole.User)
        {
            return Amazon.BedrockAgentRuntime.ConversationRole.User;
        }

        if (authorRole == AuthorRole.Assistant)
        {
            return Amazon.BedrockAgentRuntime.ConversationRole.Assistant;
        }

        throw new ArgumentOutOfRangeException($"Invalid role: {authorRole}");
    }

    #endregion
}
