#include "llama_chat_app.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  // Create chat application instance
  LlamaChatApp chatApp("models/Deep-Reasoning-Llama-3.2-Instruct-uncensored-3B.Q8_0.gguf");

  // Optional: customize configuration
  SamplingConfig samplingConfig;
  samplingConfig.temperature = 1.2f;
  samplingConfig.topP = 0.9f;
  samplingConfig.topK = 80;
  chatApp.setSamplingConfig(samplingConfig);

  // Initialize everything
  if (!chatApp.initialize())
  {
    std::cerr << "Failed to initialize chat application\n";
    return 1;
  }

  // Run the interactive chat loop
  chatApp.runChatLoop();

  return 0;
}
