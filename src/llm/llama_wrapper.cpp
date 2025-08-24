// ===== llama_wrapper.cpp =====
#include "llama_wrapper.hpp"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>

// ModelConfig implementation
ModelConfig::ModelConfig(const std::string &path) : modelPath(path)
{
  nBatch = nCtx;
}

// Constructor
LlamaWrapper::LlamaWrapper(const std::string &modelPath)
    : model(nullptr), ctx(nullptr), vocab(nullptr), sampler(nullptr),
      modelConfig(modelPath), isInitialized(false) {}

// Destructor
LlamaWrapper::~LlamaWrapper()
{
  cleanup();
}

// Configuration methods
void LlamaWrapper::setSamplingConfig(const SamplingConfig &config)
{
  samplingConfig = config;
}

void LlamaWrapper::setModelConfig(const ModelConfig &config)
{
  modelConfig = config;
}

// Initialize the model, context, and sampler
bool LlamaWrapper::initialize()
{
  if (isInitialized)
  {
    return true;
  }

  printCudaStatus();

  if (!initializeLogging())
    return false;
  if (!loadBackends())
    return false;
  if (!loadModel())
    return false;
  if (!createContext())
    return false;
  if (!setupSampler())
    return false;
  if (!setupSystemMessage(modelConfig.systemMessagePath))
    return false;

  isInitialized = true;
  return true;
}

// Main chat loop
void LlamaWrapper::runChatLoop()
{
  if (!isInitialized)
  {
    std::cerr << "Error: Must call initialize() first\n";
    return;
  }

  while (true)
  {
    printf("\033[32m> \033[0m");
    std::string userInput;
    std::getline(std::cin, userInput);

    if (userInput.empty())
    {
      break;
    }

    std::string response = processUserMessage(userInput);
    if (response.empty())
    {
      std::cerr << "Error generating response\n";
      break;
    }
  }
}

// Process a single user message and return response
std::string LlamaWrapper::processUserMessage(const std::string &userMessage)
{
  if (!isInitialized)
  {
    return "";
  }

  messageHistory.push_back({"user", strdup(userMessage.c_str())});

  std::string prompt = buildPromptFromHistory();
  if (prompt.empty())
  {
    return "";
  }

  printf("\033[33m");
  std::string response = generateResponse(prompt);
  printf("\n\033[0m");

  messageHistory.push_back({"assistant", strdup(response.c_str())});

  return response;
}

// Get current message history
const std::vector<llama_chat_message> &LlamaWrapper::getMessageHistory() const
{
  return messageHistory;
}

// Clear message history (keeps system message)
void LlamaWrapper::clearHistory()
{
  for (size_t i = 1; i < messageHistory.size(); ++i)
  {
    free(const_cast<char *>(messageHistory[i].content));
  }

  if (!messageHistory.empty())
  {
    messageHistory.resize(1);
  }
}

// Print CUDA availability status
void LlamaWrapper::printCudaStatus()
{
#ifdef GGML_USE_CUBLAS
  std::cout << "CUDA is ENABLED (GGML_USE_CUBLAS)." << std::endl;
#elif defined(LLAMA_USE_CUDA)
  std::cout << "CUDA is ENABLED (LLAMA_USE_CUDA)." << std::endl;
#else
  std::cout << "CUDA is DISABLED." << std::endl;
#endif
}

// Initialize logging to only show errors
bool LlamaWrapper::initializeLogging()
{
  llama_log_set([](enum ggml_log_level level, const char *text, void * /* user_data */)
                {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        } }, nullptr);
  return true;
}

// Load all available backends
bool LlamaWrapper::loadBackends()
{
  ggml_backend_load_all();
  return true;
}

// Load the model from file
bool LlamaWrapper::loadModel()
{
  llama_model_params modelParams = llama_model_default_params();
  modelParams.n_gpu_layers = modelConfig.nGpuLayers;

  model = llama_model_load_from_file(modelConfig.modelPath.c_str(), modelParams);
  if (!model)
  {
    fprintf(stderr, "Error: unable to load model from %s\n", modelConfig.modelPath.c_str());
    return false;
  }

  vocab = llama_model_get_vocab(model);
  return true;
}

// Create inference context
bool LlamaWrapper::createContext()
{
  llama_context_params ctxParams = llama_context_default_params();
  ctxParams.n_ctx = modelConfig.nCtx;
  ctxParams.n_batch = modelConfig.nBatch;

  ctx = llama_init_from_model(model, ctxParams);
  if (!ctx)
  {
    fprintf(stderr, "Error: failed to create llama_context\n");
    return false;
  }

  // Initialize formatted buffer
  formattedBuffer.resize(llama_n_ctx(ctx));
  return true;
}

// Setup the sampling chain
bool LlamaWrapper::setupSampler()
{
  sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

  // Add samplers in order (order matters!)
  llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
                                       samplingConfig.repetitionPenaltyLastN,
                                       samplingConfig.repetitionPenalty,
                                       0.0f, // frequency penalty disabled
                                       0.0f  // presence penalty disabled
                                       ));

  llama_sampler_chain_add(sampler, llama_sampler_init_top_k(samplingConfig.topK));
  llama_sampler_chain_add(sampler, llama_sampler_init_top_p(samplingConfig.topP, 1));
  llama_sampler_chain_add(sampler, llama_sampler_init_min_p(samplingConfig.minP, 1));
  llama_sampler_chain_add(sampler, llama_sampler_init_temp(samplingConfig.temperature));
  llama_sampler_chain_add(sampler, llama_sampler_init_dist(samplingConfig.seed));

  return true;
}

// Setup initial system message
bool LlamaWrapper::setupSystemMessage(const std::string &systemMessagePath)
{
    std::string systemMessage;
    
    if (!systemMessagePath.empty()) {
        systemMessage = readSystemMessage(systemMessagePath);
    }

    if (systemMessage.empty()) {
        systemMessage = "You are an AI assistant created by Operator Watson.\n\n"
                                   "Your purpose is to provide thoughtful, clear, and detailed responses.\n\n"
                                   "Formatting Requirements:\n\n"
                                   "1. Structure replies like this: <think>{reasoning}</think>{answer}\n"
                                   "2. <think></think> should include at least six reasoning steps when needed.\n"
                                   "3. For simple answers, <think></think> can stay empty.\n"
                                   "4. The user does not see <think></think>, so include all essential info in {answer}.\n"
                                   "5. If circular reasoning or repetition occurs, end {reasoning} with </think> and move to {answer}.\n\n"
                                   "Response Guidelines:\n\n"
                                   "1. Clear and Friendly: Use Markdown to organize and highlight points.\n"
                                   "2. Thoughtful and Precise: Explain carefully with clarity and depth.\n"
                                   "3. Reason First: Think through the problem before answering, unless trivial.\n"
                                   "4. Complete but Concise: Provide all needed details without fluff.\n"
                                   "5. Warm and Intelligent: Stay professional, approachable, and supportive.\n";
    }

  messageHistory.push_back({"system", strdup(systemMessage.c_str())});
  return true;
}

// Build prompt from current message history
std::string LlamaWrapper::buildPromptFromHistory()
{
  const char *tmpl = llama_model_chat_template(model, nullptr);

  int newLen = llama_chat_apply_template(tmpl, messageHistory.data(),
                                         messageHistory.size(), true,
                                         formattedBuffer.data(), formattedBuffer.size());

  if (newLen > static_cast<int>(formattedBuffer.size()))
  {
    formattedBuffer.resize(newLen);
    newLen = llama_chat_apply_template(tmpl, messageHistory.data(),
                                       messageHistory.size(), true,
                                       formattedBuffer.data(), formattedBuffer.size());
  }

  if (newLen < 0)
  {
    fprintf(stderr, "Failed to apply chat template\n");
    return "";
  }

  return std::string(formattedBuffer.begin(), formattedBuffer.begin() + newLen);
}

// Core generation function
std::string LlamaWrapper::generateResponse(const std::string &prompt)
{
  std::string response;

  // Determine if this is the first input
  const bool isFirst = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;

  // Tokenize prompt
  const int nPromptTokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                            nullptr, 0, isFirst, true);
  std::vector<llama_token> promptTokens(nPromptTokens);

  if (llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                     promptTokens.data(), promptTokens.size(), isFirst, true) < 0)
  {
    GGML_ABORT("Failed to tokenize prompt\n");
  }

  // Create batch and generate tokens
  llama_batch batch = llama_batch_get_one(promptTokens.data(), promptTokens.size());
  llama_token newTokenId;

  while (true)
  {
    // Check context space
    int nCtx = llama_n_ctx(ctx);
    int nCtxUsed = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;

    if (nCtxUsed + batch.n_tokens > nCtx)
    {
      printf("\033[0m\n");
      fprintf(stderr, "Context size exceeded\n");
      break;
    }

    // Run forward pass
    int ret = llama_decode(ctx, batch);
    if (ret != 0)
    {
      GGML_ABORT("Failed to decode, ret = %d\n", ret);
    }

    // Sample next token
    newTokenId = llama_sampler_sample(sampler, ctx, -1);

    // Check for end of generation
    if (llama_vocab_is_eog(vocab, newTokenId))
    {
      break;
    }

    // Convert token to text
    char buf[256];
    int n = llama_token_to_piece(vocab, newTokenId, buf, sizeof(buf), 0, true);
    if (n < 0)
    {
      GGML_ABORT("Failed to convert token to piece\n");
    }

    std::string piece(buf, n);
    printf("%s", piece.c_str());
    fflush(stdout);
    response += piece;

    // Prepare next iteration
    batch = llama_batch_get_one(&newTokenId, 1);
  }

  return response;
}

// Cleanup all allocated resources
void LlamaWrapper::cleanup()
{
  // Free message history
  for (auto &msg : messageHistory)
  {
    free(const_cast<char *>(msg.content));
  }
  messageHistory.clear();

  // Free llama.cpp resources in reverse order
  if (sampler)
  {
    llama_sampler_free(sampler);
    sampler = nullptr;
  }

  if (ctx)
  {
    llama_free(ctx);
    ctx = nullptr;
  }

  if (model)
  {
    llama_model_free(model);
    model = nullptr;
  }

  isInitialized = false;
}

// Read file contents into string
std::string LlamaWrapper::readFileContents(const std::string &filePath)
{
  std::ifstream file(filePath);
  if (!file.is_open())
  {
    std::cerr << "Error: Could not open file: " << filePath << std::endl;
    return "";
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

// Load file as first user message without generating response
bool LlamaWrapper::loadFileAsFirstMessage(const std::string &filePath)
{
  if (!isInitialized)
  {
    std::cerr << "Error: Must call initialize() first\n";
    return false;
  }

  std::string fileContent = readFileContents(filePath);
  if (fileContent.empty())
  {
    return false;
  }

  messageHistory.push_back({"user", strdup(fileContent.c_str())});
  return true;
}

// Load file as first user message and generate response
std::string LlamaWrapper::loadFileAsFirstMessageWithResponse(const std::string &filePath)
{
  if (!isInitialized)
  {
    std::cerr << "Error: Must call initialize() first\n";
    return "";
  }

  std::string fileContent = readFileContents(filePath);
  if (fileContent.empty())
  {
    return "";
  }

  // Add the file content as first user message
  messageHistory.push_back({"user", strdup(fileContent.c_str())});

  // Build prompt and generate response
  std::string prompt = buildPromptFromHistory();
  if (prompt.empty())
  {
    return "";
  }

  printf("\033[33m");
  std::string response = generateResponse(prompt);
  printf("\n\033[0m");

  // Add response to history
  messageHistory.push_back({"assistant", strdup(response.c_str())});

  return response;
}

std::string LlamaWrapper::readSystemMessage(const std::string &filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open system message file: " << filePath 
                  << ". Using default system message." << std::endl;
        return ""; // Return empty to trigger fallback
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}
