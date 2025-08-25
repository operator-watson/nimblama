#include "llama_wrapper.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  // Create chat application instance
  LlamaWrapper lw("models/l3.1-dark-reasoning-lewdplay-evo-hermes-r1-uncensored-8b-q4_k_m.gguf");

  // Optional: customize configuration
  SamplingConfig samplingConfig;
  samplingConfig.temperature = 1.2f;
  samplingConfig.topP = 0.9f;
  samplingConfig.topK = 80;
  lw.setSamplingConfig(samplingConfig);

  ModelConfig modelConfig("models/l3.1-dark-reasoning-lewdplay-evo-hermes-r1-uncensored-8b-q4_k_m.gguf");
  modelConfig.systemMessagePath = "system_message.txt";
  modelConfig.nGpuLayers = 80;
  modelConfig.nCtx = 12288;
  modelConfig.nBatch = 2048;
  lw.setModelConfig(modelConfig);

  lw.enableChatLogging(true, "chat_logs");

  // Initialize everything
  if (!lw.initialize())
  {
    std::cerr << "Failed to initialize chat application\n";
    return 1;
  }

    // Run the interactive chat loop
    lw.runChatLoop();

/*   std::string response = lw.loadFileAsFirstMessageWithResponse("prompt.txt");
  if (!response.empty())
  {
    // File was processed and response generated
    lw.runChatLoop(); // Continue chatting about the file
  } */

  return 0;
}