// ===== llama_wrapper.h =====
#pragma once

#include "llama.h"
#include <string>
#include <vector>

// Configuration structure for sampling parameters
struct SamplingConfig
{
  float temperature = 1.2f;
  float topP = 0.9f;
  int topK = 80;
  float minP = 0.02f;
  float repetitionPenalty = 1.05f;
  int repetitionPenaltyLastN = 128;
  uint32_t seed = LLAMA_DEFAULT_SEED;
};

// Configuration structure for model and context parameters
struct ModelConfig
{
  std::string modelPath;
  int nGpuLayers = 100;
  int nCtx = 8192;
  int nBatch = 8192;

  explicit ModelConfig(const std::string &path);
};

// Main chat application class
class LlamaWrapper
{
private:
  llama_model *model;
  llama_context *ctx;
  const llama_vocab *vocab;
  llama_sampler *sampler;

  std::vector<llama_chat_message> messageHistory;
  std::vector<char> formattedBuffer;

  ModelConfig modelConfig;
  SamplingConfig samplingConfig;

  bool isInitialized;

public:
  explicit LlamaWrapper(const std::string &modelPath);
  ~LlamaWrapper();

  // Configuration methods
  void setSamplingConfig(const SamplingConfig &config);
  void setModelConfig(const ModelConfig &config);

  // Core functionality
  bool initialize();
  void runChatLoop();
  std::string processUserMessage(const std::string &userMessage);

  // Utility methods
  const std::vector<llama_chat_message> &getMessageHistory() const;
  void clearHistory();

private:
  // Initialization helpers
  void printCudaStatus();
  bool initializeLogging();
  bool loadBackends();
  bool loadModel();
  bool createContext();
  bool setupSampler();
  bool setupSystemMessage();

  // Generation helpers
  std::string buildPromptFromHistory();
  std::string generateResponse(const std::string &prompt);

  // Resource management
  void cleanup();
};
