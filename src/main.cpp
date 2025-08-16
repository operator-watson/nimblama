#include "llama_wrapper.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  // Create chat application instance
  LlamaWrapper lw("models/Deep-Reasoning-Llama-3.2-Instruct-uncensored-3B.Q8_0.gguf");

  // Optional: customize configuration
  SamplingConfig samplingConfig;
  samplingConfig.temperature = 1.2f;
  samplingConfig.topP = 0.9f;
  samplingConfig.topK = 80;
  lw.setSamplingConfig(samplingConfig);

  ModelConfig modelConfig("models/Deep-Reasoning-Llama-3.2-Instruct-uncensored-3B.f16.gguf");
  modelConfig.nCtx = 24576;
  modelConfig.nBatch = 2048;
  lw.setModelConfig(modelConfig);

  // Initialize everything
  if (!lw.initialize())
  {
    std::cerr << "Failed to initialize chat application\n";
    return 1;
  }

  /*   // Run the interactive chat loop
    lw.runChatLoop(); */

  std::string response = lw.loadFileAsFirstMessageWithResponse("prompt.txt");
  if (!response.empty())
  {
    // File was processed and response generated
    lw.runChatLoop(); // Continue chatting about the file
  }

  return 0;
}
