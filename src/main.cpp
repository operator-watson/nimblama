#include "llama_wrapper.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  // Create chat application instance
  LlamaWrapper lw("models/L3.1-DpHermes-Rsnng-8B-DarkIdol-Inst-1.2-Uncen-D_AU-Q4_k_m.gguf");

  // Optional: customize configuration
  SamplingConfig samplingConfig;
  samplingConfig.temperature = 1.2f;
  samplingConfig.topP = 0.9f;
  samplingConfig.topK = 80;
  lw.setSamplingConfig(samplingConfig);

  ModelConfig modelConfig("models/L3.1-DpHermes-Rsnng-8B-DarkIdol-Inst-1.2-Uncen-D_AU-Q4_k_m.gguf");
  modelConfig.systemMessagePath = "";
  modelConfig.nGpuLayers = 100;
  modelConfig.nCtx = 16384;
  modelConfig.nBatch = 4096;
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