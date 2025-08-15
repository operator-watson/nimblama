#include "llama.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

void print_cuda_status()
{
#ifdef GGML_USE_CUBLAS
  std::cout << "CUDA is ENABLED (GGML_USE_CUBLAS)." << std::endl;
#elif defined(LLAMA_USE_CUDA)
  std::cout << "CUDA is ENABLED (LLAMA_USE_CUDA)." << std::endl;
#else
  std::cout << "CUDA is DISABLED." << std::endl;
#endif
}

int main(int argc, char **argv)
{
  print_cuda_status();
  // Path to the GGUF model file
  std::string model_path = "models/Deep-Reasoning-Llama-3.2-Instruct-uncensored-3B.Q8_0.gguf";

  // Number of layers to offload to the GPU (ngl = number of GPU layers)
  int ngl = 100;

  // Context length for the model (how much history the model remembers in one go)
  int n_ctx = 8192;

  // Set up a logging function to only print error messages from llama.cpp internals
  llama_log_set([](enum ggml_log_level level, const char *text, void * /* user_data */)
                {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        } }, nullptr);

  // Load all available backends (CPU, CUDA, etc.) from llama.cpp/ggml
  ggml_backend_load_all();

  // Set up model loading parameters
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = ngl; // How many transformer layers to offload to GPU

  // Load the model from file using the parameters above
  llama_model *model = llama_model_load_from_file(model_path.c_str(), model_params);
  if (!model)
  {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return 1;
  }

  // Get the vocabulary object (token-to-string mapping) from the model
  const llama_vocab *vocab = llama_model_get_vocab(model);

  // Set up context parameters (used during inference)
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = n_ctx;   // Maximum tokens context can store
  ctx_params.n_batch = n_ctx; // Batch size, often equals context size

  // Create an inference context (used to actually run the model)
  llama_context *ctx = llama_init_from_model(model, ctx_params);
  if (!ctx)
  {
    fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
    return 1;
  }

  // Create sampler chain with configurable parameters
  float temperature = 1.2f;            // Higher for more creativity and unexpected word choices
  float top_p = 0.9f;                  // Slightly lower to avoid completely random tokens
  int top_k = 80;                      // Higher to consider more vocabulary options
  float min_p = 0.02f;                 // Lower to allow more diverse token selection
  float repetition_penalty = 1.05f;    // Lighter penalty - some repetition can be stylistic in stories
  int repetition_penalty_last_n = 128; // Look back further to avoid repetitive phrases/patterns

  llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());

  // Add samplers in order (order matters!)
  llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                                    repetition_penalty_last_n, // last_n tokens to consider
                                    repetition_penalty,        // repeat penalty
                                    0.0f,                      // frequency penalty (0.0 = disabled)
                                    0.0f));

  llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
  llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
  llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
  llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
  llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  // Lambda function that takes a prompt and returns a response by sampling from the model
  auto generate = [&](const std::string &prompt)
  {
    std::string response;

    // Determine whether this is the first input (needed for some tokenizer modes)
    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;

    // Tokenize the prompt to convert it to llama_token array
    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0)
    {
      GGML_ABORT("failed to tokenize the prompt\n");
    }

    // Build a batch object from tokens to feed into the model
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;

    while (true)
    {
      // Ensure we have enough space in the context for new tokens
      int n_ctx = llama_n_ctx(ctx);
      int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
      if (n_ctx_used + batch.n_tokens > n_ctx)
      {
        printf("\033[0m\n");
        fprintf(stderr, "context size exceeded\n");
        exit(0);
      }

      // Run model forward pass on batch
      int ret = llama_decode(ctx, batch);
      if (ret != 0)
      {
        GGML_ABORT("failed to decode, ret = %d\n", ret);
      }

      // Sample next token using the configured sampler
      new_token_id = llama_sampler_sample(smpl, ctx, -1);

      // Break the loop if EOS (end of sequence) token is reached
      if (llama_vocab_is_eog(vocab, new_token_id))
      {
        break;
      }

      // Convert token back to readable string
      char buf[256];
      int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
      if (n < 0)
      {
        GGML_ABORT("failed to convert token to piece\n");
      }
      std::string piece(buf, n);
      printf("%s", piece.c_str()); // Output the generated token
      fflush(stdout);
      response += piece; // Append to response

      // Prepare next token for continuation
      batch = llama_batch_get_one(&new_token_id, 1);
    }

    return response;
  };

  // Store the message history for chat context
  std::vector<llama_chat_message> messages;

  // Allocate a buffer for storing the templated input for the model
  std::vector<char> formatted(llama_n_ctx(ctx));

  // Add a system message to guide the assistant's behavior
  messages.push_back({"system",
                      strdup("You are an AI assistant created by Operator Watson.\n\nYour purpose is to provide thoughtful, clear, and detailed responses.\n\nFormatting Requirements:\n\n1. Structure replies like this: <think>{reasoning}</think>{answer}\n2. <think></think> should include at least six reasoning steps when needed.\n3. For simple answers, <think></think> can stay empty.\n4. The user does not see <think></think>, so include all essential info in {answer}.\n5. If circular reasoning or repetition occurs, end {reasoning} with </think> and move to {answer}.\n\nResponse Guidelines:\n\n1. Clear and Friendly: Use Markdown to organize and highlight points.\n2. Thoughtful and Precise: Explain carefully with clarity and depth.\n3. Reason First: Think through the problem before answering, unless trivial.\n4. Complete but Concise: Provide all needed details without fluff.\n5. Warm and Intelligent: Stay professional, approachable, and supportive.")});

  int prev_len = 0; // Tracks where the previous message ended in the buffer

  while (true)
  {
    // Prompt the user for input
    printf("\033[32m> \033[0m");
    std::string user;
    std::getline(std::cin, user);

    // Exit loop on empty input
    if (user.empty())
    {
      break;
    }

    // Get the chat template for the model (defines role formatting, e.g., "<user>")
    const char *tmpl = llama_model_chat_template(model, /* name */ nullptr);

    // Add the new user message to the message history
    messages.push_back({"user", strdup(user.c_str())});

    // Apply chat template to full history to build the new prompt
    int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    if (new_len > (int)formatted.size())
    {
      // Resize buffer and retry if needed
      formatted.resize(new_len);
      new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    }
    if (new_len < 0)
    {
      fprintf(stderr, "failed to apply the chat template\n");
      return 1;
    }

    // Extract only the new part of the prompt (user message)
    std::string prompt(formatted.begin(), formatted.begin() + new_len);

    // Generate the assistant's response
    printf("\033[33m");
    std::string response = generate(prompt);
    printf("\n\033[0m");

    // Add the assistant's reply to message history
    messages.push_back({"assistant", strdup(response.c_str())});

    // Update `prev_len` to where this round of message ends
    prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
    if (prev_len < 0)
    {
      fprintf(stderr, "failed to apply the chat template\n");
      return 1;
    }
  }

  // Clean up allocated message memory
  for (auto &msg : messages)
  {
    free(const_cast<char *>(msg.content));
  }

  // Free all resources in reverse order of creation
  llama_sampler_free(smpl);
  llama_free(ctx);
  llama_model_free(model);

  return 0;
}
