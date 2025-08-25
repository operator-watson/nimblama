// ===== llama_wrapper.h =====
#pragma once

#include "llama.h"
#include <string>
#include <vector>
#include <fstream>

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
  std::string systemMessagePath = "";
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

  bool loggingEnabled = false;
  std::string logDirectory = "chat_logs";
  std::ofstream logFile;
  std::string currentLogPath;

public:
  explicit LlamaWrapper(const std::string &modelPath);
  ~LlamaWrapper();

  // Configuration methods
  void setSamplingConfig(const SamplingConfig &config);
  void setModelConfig(const ModelConfig &config);

  void enableChatLogging(bool enable = true, const std::string& directory = "chat_logs");
  std::string getCurrentLogFilePath() const { return currentLogPath; }

  // Core functionality
  bool initialize();
  void runChatLoop();
  std::string processUserMessage(const std::string &userMessage);

  // NEW: File loading functionality
  bool loadFileAsFirstMessage(const std::string &filePath);
  std::string loadFileAsFirstMessageWithResponse(const std::string &filePath);

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
  bool setupSystemMessage(const std::string &systemMessagePath = "");

  // Generation helpers
  std::string buildPromptFromHistory();
  std::string generateResponse(const std::string &prompt);

  // Resource management
  void cleanup();

  // NEW: File reading helper
  std::string readFileContents(const std::string &filePath);
  std::string readSystemMessage(const std::string &filePath);

  std::string generateLogFilename();
  void writeToLog(const std::string& role, const std::string& content);
  bool createLogFile();
};
