#include "datasets/uci_adult_income/load_uci_adult_income.h"
#include "ops/rand_fill/rand_fill.h"
#include "modules/DCNEmbedding.h"
#include "modules/DCN.h"
#include "optim/SGD.h"
#include "ops/bce_logits_loss/bce_logits_loss.h"
#include "modules/MLP.h"


void dcn() {
    UCIAdultIncome data = load_uci_adult_income();

    DCNEmbedding embedding(data.category_sizes, 8);
    DCN dcn(data.dense.size(1) + data.category.size(1) * 8, 3, {64, 32}, 1);

    auto params = embedding.parameters();
    auto dcn_params = dcn.parameters();
    params.insert(params.end(), dcn_params.begin(), dcn_params.end());
    SGD sgd(params);

    int block_size = 100;

    for (int i = 0; i < data.dense.size(0);) {
        float total_loss = 0.0f;
        int correct = 0;
        int pos_pred = 0;
        sgd.zero_grad();
        for (int j = 0; j < block_size && i < data.dense.size(0); j++, i++) {

            Tensor dense = data.dense.slice(0, i, i+1);
            Tensor cat = data.category.slice(0, i, i+1);

            Tensor x = embedding.forward({cat, dense});
            
            Tensor out = dcn.forward({x}).view(Shape({1}));
            Tensor label = data.labels.slice(0, i, i+1);


            Tensor loss = bce_logits_loss(out, label);
            loss.backward();

            total_loss += loss.data()[0];


            if ((out.data()[0] > 0) == (label.data()[0] > 0.5f)) {
                correct++;
            }

            if (out.data()[0] > 0) {
                pos_pred++;
            }
        }

        
        sgd.step(0.0001f);

        std::cout << "Avg Loss: " << total_loss / block_size << std::endl;
        std::cout << "Correct rate: " << correct << std::endl;
        std::cout << "Positive predictions: " << pos_pred << std::endl;

    }
}