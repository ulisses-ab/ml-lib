#include <iostream>
#include <vector>
#include <cmath>
#include "modules/GPT.h"
#include "optim/SGD.h"
#include "optim/Adam.h"
#include "ops/cross_entropy/cross_entropy.h"
#include "datasets/simple_english/load_simple_english.h"

using namespace std;

void transformer() {
    std::cout << "Loading SimpleEnglish dataset..." << endl;
    SimpleEnglish wiki = load_simple_english(30000);
    Tensor dataset = wiki.data;
    int vocab_size = wiki.vocab_size;
    
    std::cout << "Dataset loaded. Vocab size: " << vocab_size << endl;
    std::cout << "Total tokens: " << dataset.numel() << endl;
    
    const int d_model = 128;
    const int num_heads = 4;
    const int num_layers = 2;
    const int d_ff = 512;
    const int context_len = 64; 
    const float learning_rate = 1e-4f;  
    const int num_epochs = 1;
    
    std::cout << "Creating GPT model..." << endl;
    GPT model(vocab_size, d_model, num_heads, num_layers, d_ff, context_len, Device::CUDA);
    
    auto params = model.parameters();
    Adam adam(params);
    
    std::cout << "Starting training..." << endl;
    std::cout << "Context length: " << context_len << endl;
    std::cout << "Learning rate: " << learning_rate << endl;
    
    int total_tokens = dataset.numel();
    int num_sequences = total_tokens - context_len;  
    
    float total_loss = 0.0f;
    int step_count = 0;
    const int log_interval = 30;  
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << "\nEpoch " << epoch << endl;
        
        for (int i = 0; i < num_sequences; i++) {
            adam.zero_grad();
            
            Tensor input_seq = dataset.slice(0, i, i + context_len);
            Tensor target_seq = dataset.slice(0, i + 1, i + context_len + 1);

            input_seq = input_seq.copy_to(Device::CUDA);
            Tensor logits = model.forward_logits({input_seq}).copy_to(Device::CPU);

            Tensor loss = cross_entropy(logits, target_seq);
            float loss_value = loss.data()[0];
            
            total_loss += loss_value;
            step_count++;
            
            loss.backward();

            adam.step(learning_rate);

            if (step_count % log_interval == 0) {
                const int T = context_len;
                const int V = vocab_size;

                std::cout << "Input: ";
                Tensor input_snippet = input_seq.slice(0, input_seq.size(0)-30, input_seq.size(0)).copy_to(Device::CPU);
                std::cout << decode_simple_english(input_snippet) << " ";
                std::cout << endl;

                float* last_logits = logits.data() + (T - 1) * V;

                int predicted_token = 0;
                float best = last_logits[0];
                for (int v = 1; v < V; v++) {
                    if (last_logits[v] > best) {
                        best = last_logits[v];
                        predicted_token = v;
                    }
                }

                Tensor predict(Shape{{1}});
                predict.at({0}) = predicted_token;
                std::cout << "Predicted next token: " << decode_simple_english(predict) << endl;

                float avg_total_loss = total_loss / step_count;
                std::cout << "Step " << step_count
                    << " | Loss: " << loss_value
                    << " | Avg Loss: " << avg_total_loss
                    << " | Progress: " << (100.0f * i / num_sequences) << "%"
                    << endl << endl;
            }

        }
        
        float epoch_avg_loss = total_loss / step_count;
        std::cout << "\nEpoch " << epoch << " completed. Average loss: " << epoch_avg_loss << endl;
    }
    
    std::cout << "\nTraining completed!" << endl;
}
