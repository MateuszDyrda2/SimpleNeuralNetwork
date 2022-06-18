#include <biai/network2.h>
#include <iostream>

using namespace biai;
int main(void)
{
#if 1
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
                  .hidden_layer(2)
                  .output_layer(1)
                  .learn_stochastic(true)
                  .learning_rate(0.1)
                  .training_momentum(0.9)
                  .max_epochs(10000)
                  .build();

    nn->train(entries);

    std::cout << "Predicting for examples\n";
    auto res = nn->predict({ 0.f, 0.f });
    std::cout << "for [0,0] -> " << res[0] << '\n';

    auto res2 = nn->predict({ 1.f, 0.f });
    std::cout << "for [1,0] -> " << res2[0] << '\n';

    auto res3 = nn->predict({ 0.f, 1.f });
    std::cout << "for [0,1] -> " << res3[0] << '\n';

    auto res4 = nn->predict({ 1.f, 1.f });
    std::cout << "for [1,1] -> " << res4[0] << '\n';

#endif
#if 0
    std::vector<dataset::entry_t> entries;
    dataset::entry_t entry{ { 0.f, 0.f }, { 0.f } };
    dataset::entry_t entry2{ { 1.f, 0.f }, { 1.f } };
    dataset::entry_t entry3{ { 0.f, 1.f }, { 1.f } };
    dataset::entry_t entry4{ { 1.f, 1.f }, { 1.f } };
    entries.push_back(entry);
    entries.push_back(entry2);
    entries.push_back(entry3);
    entries.push_back(entry4);

    auto nn = neural_network::create()
                  .input_layer(2)
                  .output_layer(1)
                  .learn_stochastic(false)
                  .learning_rate(0.1)
                  .max_epochs(10000)
                  .build();

    nn->train(entries);

    auto res = nn->predict({ 0.f, 0.f });
    std::cout << "for [0,0] -> " << res[0] << '\n';
    auto res1 = nn->predict({ 0.f, 1.f });
    std::cout << "for [0,1] -> " << res1[0] << '\n';
    auto res2 = nn->predict({ 1.f, 0.f });
    std::cout << "for [1,0] -> " << res2[0] << '\n';
    auto res3 = nn->predict({ 1.f, 1.f });
    std::cout << "for [1,1] -> " << res3[0] << '\n';

#endif

    return 0;
}