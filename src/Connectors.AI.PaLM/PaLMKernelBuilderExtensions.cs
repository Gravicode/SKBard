// Copyright (c) Microsoft. All rights reserved.

using System.Net.Http;
using Connectors.AI.PaLM.TextEmbedding;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI.Embeddings;
using Microsoft.SemanticKernel.AI.TextCompletion;
using Microsoft.SemanticKernel.Connectors.AI.PaLM.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AI.PaLM.TextCompletion;

#pragma warning disable IDE0130
// ReSharper disable once CheckNamespace - Using NS of KernelConfig
namespace Microsoft.SemanticKernel;
#pragma warning restore IDE0130

/// <summary>
/// Provides extension methods for the <see cref="KernelBuilder"/> class to configure PaLM connectors.
/// </summary>
public static class PaLMKernelBuilderExtensions
{
    /// <summary>
    /// Registers an PaLM text completion service with the specified configuration.
    /// </summary>
    /// <param name="builder">The <see cref="KernelBuilder"/> instance.</param>
    /// <param name="model">The name of the PaLM model.</param>
    /// <param name="apiKey">The API key required for accessing the PaLM service.</param>
    /// <param name="endpoint">The endpoint URL for the text completion service.</param>
    /// <param name="serviceId">A local identifier for the given AI service.</param>
    /// <param name="httpClient">The optional <see cref="HttpClient"/> to be used for making HTTP requests.
    /// If not provided, a default <see cref="HttpClient"/> instance will be used.</param>
    /// <returns>The modified <see cref="KernelBuilder"/> instance.</returns>
    public static KernelBuilder WithPaLMTextCompletionService(this KernelBuilder builder,
        string model,
        string? apiKey = null,
        string? endpoint = null,
        string? serviceId = null,
        HttpClient? httpClient = null)
    {
        builder.WithAIService<ITextCompletion>(serviceId, (parameters) =>
            new PaLMTextCompletion(
                model,
                apiKey,
                new HttpClient(),
                endpoint));

        return builder;
    }


    public static KernelBuilder WithPaLMChatCompletionService(this KernelBuilder builder,
        string model,
        string apiKey = null, string? serviceId=null)
    {
        builder.WithAIService<PaLMChatCompletion>(serviceId, (parameters) =>
            new PaLMChatCompletion(
                model,
                apiKey));

        return builder;
    }
   

    /// <summary>
    /// Registers an PaLM text embedding generation service with the specified configuration.
    /// </summary>
    /// <param name="builder">The <see cref="KernelBuilder"/> instance.</param>
    /// <param name="model">The name of the PaLM model.</param>
    /// <param name="apiKey">API Key for PaLM.</param>
    /// <param name="serviceId">A local identifier for the given AI service.</param>
    /// <returns>The <see cref="KernelBuilder"/> instance.</returns>
    public static KernelBuilder WithPaLMTextEmbeddingGenerationService(this KernelBuilder builder,
        string model,
        string apiKey,
        string? serviceId = null)
    {
        builder.WithAIService<ITextEmbeddingGeneration>(serviceId, (parameters) =>
            new PaLMTextEmbeddingGeneration(
                model,
                apiKey: apiKey));

        return builder;
    }

    /// <summary>
    /// Registers an PaLM text embedding generation service with the specified configuration.
    /// </summary>
    /// <param name="builder">The <see cref="KernelBuilder"/> instance.</param>
    /// <param name="model">The name of the PaLM model.</param>
    /// <param name="httpClient">The optional <see cref="HttpClient"/> instance used for making HTTP requests.</param>
    /// <param name="endpoint">The endpoint for the text embedding generation service.</param>
    /// <param name="serviceId">A local identifier for the given AI serviceю</param>
    /// <returns>The <see cref="KernelBuilder"/> instance.</returns>
    public static KernelBuilder WithPaLMTextEmbeddingGenerationService(this KernelBuilder builder,
        string model,
        HttpClient? httpClient = null,
        string? endpoint = null,
        string? serviceId = null)
    {
        builder.WithAIService<ITextEmbeddingGeneration>(serviceId, (parameters) =>
            new PaLMTextEmbeddingGeneration(
                model,
                new HttpClient(),
                endpoint));

        return builder;
    }
}
