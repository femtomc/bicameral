# Running Bicamrl with LM Studio

This guide explains how to use LM Studio for local LLM inference with Bicamrl's production tests and Sleep Layer features.

## Prerequisites

1. **Install LM Studio**: Download from [https://lmstudio.ai/](https://lmstudio.ai/)
2. **Download a Model**: In LM Studio, download a model like:
   - `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
   - `TheBloke/CodeLlama-13B-Instruct-GGUF`
   - `TheBloke/Llama-2-7B-Chat-GGUF`

## Setup Steps

### 1. Start LM Studio Server

1. Open LM Studio
2. Load your chosen model
3. Go to the "Local Server" tab
4. Click "Start Server" (default port: 1234)
5. Note the model name shown in the server info

### 2. Run Production Test

Pass the model name shown in LM Studio as an argument:

```bash
# Quick test with LM Studio
pixi run test-lmstudio "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

# Full production test with LM Studio
pixi run test-lmstudio-full "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
```

Replace `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` with your actual model name from LM Studio.

### 3. (Optional) Customize Configuration

The commands above automatically configure the model name. If you need to change other settings (port, temperature, etc.):

```bash
# Edit the config after it's created
nano ~/.bicamrl/Mind.toml
```

## Configuration Options

### Using Different Ports

If LM Studio is running on a different port:

```toml
[sleep.llm_providers.lmstudio]
base_url = "http://localhost:8080/v1"  # Your custom port
```

### Adjusting for Model Capabilities

Different models have different context windows and capabilities:

```toml
[sleep.llm_providers.lmstudio]
max_tokens = 1024  # Smaller for 7B models
temperature = 0.5  # Lower for more consistent output
```

### Performance Tuning

For faster responses with local models:

```toml
[sleep]
batch_size = 3  # Smaller batches
analysis_interval = 600  # Less frequent analysis
```

## Recommended Models

### For Code Tasks
- **CodeLlama-13B-Instruct**: Best for code analysis and generation
- **Deepseek-Coder-6.7B**: Optimized for programming tasks

### For General Analysis
- **Mistral-7B-Instruct**: Good balance of speed and quality
- **Llama-2-13B-Chat**: Better reasoning but slower

### For Fast Testing
- **Phi-2**: Very fast, good for quick tests
- **TinyLlama-1.1B**: Extremely fast but limited capabilities

## Troubleshooting

### Connection Refused
- Ensure LM Studio server is running
- Check the port number matches your config
- Try `curl http://localhost:1234/v1/models` to test

### Slow Responses
- Use a smaller model
- Reduce `max_tokens` in config
- Enable GPU acceleration in LM Studio

### Poor Quality Results
- Try a larger model (13B+)
- Adjust temperature (0.3-0.7 for analysis tasks)
- Use models specifically trained for instructions

## Environment Variables

You can also use environment variables:

```bash
export OPENAI_BASE_URL="http://localhost:1234/v1"
export OPENAI_API_KEY="not-needed"
pixi run test-production
```

## Limitations

When using local models instead of OpenAI/Claude:

1. **Quality**: Local models may produce less accurate analysis
2. **Speed**: Larger models can be slow without GPU
3. **Context**: Most local models have smaller context windows
4. **JSON Output**: Some models struggle with structured output

For best results with production testing, use models specifically fine-tuned for instruction following and code tasks.
