// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Connectors.AI.PaLM;
using Connectors.AI.PaLM.TextCompletion;
using Microsoft.SemanticKernel.AI;
using Microsoft.SemanticKernel.AI.TextCompletion;
using Microsoft.SemanticKernel.Diagnostics;
using Microsoft.SemanticKernel.Services;

namespace Microsoft.SemanticKernel.Connectors.AI.PaLM.TextCompletion;

/// <summary>
/// PaLM text completion service.
/// </summary>
public sealed class PaLMTextCompletion : ITextCompletion,IDisposable
{
    private const string HttpUserAgent = "Microsoft-Semantic-Kernel";
    private const string PaLMApiEndpoint = "https://generativelanguage.googleapis.com/v1beta2/models";

    private readonly string _model = "text-bison-001";
    private readonly string? _endpoint;
    private readonly HttpClient _httpClient;
    private readonly string? _apiKey;

    private readonly Dictionary<string, string> _attributes = new();

    public IReadOnlyDictionary<string, string> Attributes => _attributes;

    public void Dispose()
    {
        _httpClient.Dispose();
    }
    /// <summary>
    /// Initializes a new instance of the <see cref="PaLMTextCompletion"/> class.
    /// Using default <see cref="HttpClientHandler"/> implementation.
    /// </summary>
    /// <param name="endpoint">Endpoint for service API call.</param>
    /// <param name="model">Model to use for service API call.</param>
    public PaLMTextCompletion(Uri endpoint, string model)
    {
        VerifyHelper.NotNull(endpoint);
        VerifyHelper.NotNullOrWhiteSpace(model);

        this._endpoint = endpoint.AbsoluteUri;
        this._model = model;

        this._httpClient = new HttpClient();
        this._attributes.Add(IAIServiceExtensions.ModelIdKey, this._model);
        this._attributes.Add(IAIServiceExtensions.EndpointKey, this._endpoint);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PaLMTextCompletion"/> class.
    /// Using PaLM API for service call, see https://developers.generativeai.google/guide/palm_api_overview.
    /// </summary>
    /// <param name="model">The name of the model to use for text completion.</param>
    /// <param name="apiKey">The API key for accessing the PaLM service.</param>
    /// <param name="httpClient">The HTTP client to use for making API requests. If not specified, a default client will be used.</param>
    /// <param name="endpoint">The endpoint URL for the PaLM service.
    /// If not specified, the base address of the HTTP client is used. If the base address is not available, a default endpoint will be used.</param>
    public PaLMTextCompletion(string model, string? apiKey = null, HttpClient? httpClient = null, string? endpoint = null)
    {
        VerifyHelper.NotNullOrWhiteSpace(model);
        VerifyHelper.NotNullOrWhiteSpace(apiKey);

        this._model = model;
        this._apiKey = apiKey;
        this._httpClient = httpClient ?? new HttpClient();
        this._endpoint = endpoint;
        this._attributes.Add(IAIServiceExtensions.ModelIdKey, this._model);
        this._attributes.Add(IAIServiceExtensions.EndpointKey, this._endpoint ?? PaLMApiEndpoint);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<ITextStreamingResult> GetStreamingCompletionsAsync(
        string text,
        AIRequestSettings requestSettings,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (var completion in await this.ExecuteGetCompletionsAsync(text, cancellationToken).ConfigureAwait(false))
        {
            yield return completion;
        }
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ITextResult>> GetCompletionsAsync(
        string text,
        AIRequestSettings requestSettings,
        CancellationToken cancellationToken = default)
    {
       return await this.ExecuteGetCompletionsAsync2(text, cancellationToken).ConfigureAwait(false);
       
    }

    #region private ================================================================================
    private async Task<IReadOnlyList<ITextResult>> ExecuteGetCompletionsAsync2(string text, CancellationToken cancellationToken = default)
    {
        try
        {
            var completionRequest = new TextCompletionRequest();
            completionRequest.Prompt.Text = text;

            using var httpRequestMessage = new HttpRequestMessage()
            {
                Method = HttpMethod.Post,
                RequestUri = this.GetRequestUri(),
                Content = new StringContent(JsonSerializer.Serialize(completionRequest)),
            };

            httpRequestMessage.Headers.Add("User-Agent", HttpUserAgent);

            using var response = await this._httpClient.SendAsync(httpRequestMessage, cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);

            TextCompletionResponse? completionResponse = JsonSerializer.Deserialize<TextCompletionResponse>(body);
            //List<TextCompletionResponse>? completionResponse = JsonSerializer.Deserialize<List<TextCompletionResponse>>(body);

            if (completionResponse is null)
            {
                throw new SKException("Unexpected response from model")
                {
                    Data = { { "ResponseData", body } },
                };
            }

            //note: if PaLM refuse to answer it will response with different json schema, without any candidates
            if (completionResponse.Candidates is null)
            {
                var errorResponse = JsonSerializer.Deserialize<TextCompletionError>(body);
                var reason = errorResponse?.Filters.First()?.Reason;
                throw new SKException($"Unexpected response from model: {reason}")
                {
                    Data = { { "Reason", reason } }
                };

            }

            return new List<TextCompletionResult>() { new TextCompletionResult(completionResponse) };

        }
        catch (Exception e)
        {
            throw new SKException(
                $"Something went wrong: {e.Message}", e);
        }
    }
    private async Task<IReadOnlyList<ITextStreamingResult>> ExecuteGetCompletionsAsync(string text, CancellationToken cancellationToken = default)
    {
        try
        {
            var completionRequest = new TextCompletionRequest();
            completionRequest.Prompt.Text = text;

            using var httpRequestMessage = new HttpRequestMessage()
            {
                Method = HttpMethod.Post,
                RequestUri = this.GetRequestUri(),
                Content = new StringContent(JsonSerializer.Serialize(completionRequest)),
            };

            httpRequestMessage.Headers.Add("User-Agent", HttpUserAgent);
           
            using var response = await this._httpClient.SendAsync(httpRequestMessage, cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);

            TextCompletionResponse? completionResponse = JsonSerializer.Deserialize<TextCompletionResponse>(body);

            if (completionResponse is null)
            {
                throw new SKException( "Unexpected response from model")
                {
                    Data = { { "ResponseData", body } },
                };
            }

            //note: if PaLM refuse to answer it will response with different json schema, without any candidates
            if (completionResponse.Candidates is null)
            {
                var errorResponse = JsonSerializer.Deserialize<TextCompletionError>(body);
                throw new SKException( "Unexpected response from model")
                {
                    Data = { { "Reason", errorResponse?.Filters.First()?.Reason } }
                };
               
            }

            //return completionResponse.ConvertAll(c => new TextCompletionStreamingResult(c));
            return new List<ITextStreamingResult>() { new TextCompletionStreamingResult(completionResponse) };
        }
        catch (Exception e) 
        {
            throw new SKException(
                $"Something went wrong: {e.Message}", e);
        }
    }
    
    /// <summary>
    /// Retrieves the request URI based on the provided endpoint and model information.
    /// </summary>
    /// <returns>
    /// A <see cref="Uri"/> object representing the request URI.
    /// </returns>
    private Uri GetRequestUri()
    {
        var baseUrl = PaLMApiEndpoint;

        if (!string.IsNullOrEmpty(this._endpoint))
        {
            baseUrl = this._endpoint;
        }
        else if (this._httpClient.BaseAddress?.AbsoluteUri != null)
        {
            baseUrl = this._httpClient.BaseAddress!.AbsoluteUri;
        }

        var url = $"{baseUrl!.TrimEnd('/')}/{this._model}:generateText?key={this._apiKey}";

        return new Uri(url);
    }

    #endregion
}
