#include <biai/network2.h>
#include <iostream>

using namespace biai;
int main(void)
{
    std::vector<dataset::entry_t> entries;
    dataset::entry_t entry;

    entry.first  = neural_network::vector_t{ 0.f, 0.f };
    entry.second = neural_network::vector_t{ 0.f };
    entries.push_back(entry);
    entry.first  = neural_network::vector_t{ 0.f, 1.f };
    entry.second = neural_network::vector_t{ 1.f };
    entries.push_back(entry);
    entry.first  = neural_network::vector_t{ 1.f, 0.f };
    entry.second = neural_network::vector_t{ 1.f };
    entries.push_back(entry);
    entry.first  = neural_network::vector_t{ 1.f, 1.f };
    entry.second = neural_network::vector_t{ 0.f };
    entries.push_back(entry);

    auto nn = neural_network::create()
                  .input_layer(2)
                  .hidden_layer(3)
                  .output_layer(1)
                  .learn_stochastic(true)
                  .learning_rate(0.1)
                  .max_epochs(1000)
                  .build();

    nn->train(entries);

    std::cout << "Predicting for examples\n";
    auto res = nn->predict(neural_network::vector_t{ 0.f, 0.f });
    std::cout << "for [0,0] -> " << res[0] << '\n';

    auto res2 = nn->predict(neural_network::vector_t{ 1.f, 0.f });
    std::cout << "for [1,0] -> " << res[0] << '\n';

    auto res3 = nn->predict(neural_network::vector_t{ 0.f, 1.f });
    std::cout << "for [0,1] -> " << res[0] << '\n';

    auto res4 = nn->predict(neural_network::vector_t{ 1.f, 1.f });
    std::cout << "for [1,1] -> " << res[0] << '\n';

    return 0;
}