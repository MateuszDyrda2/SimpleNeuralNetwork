#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace biai {
// TODO:	implement batch vs mini batch vs stochastic gradient descent
class neural_network
{
  public:
    template<class T>
    using array_t   = std::vector<T>;
    using self_type = neural_network;
    using size_type = std::size_t;

  public:
    class network_builder
    {
      public:
        network_builder& input_layer(std::size_t size);
        network_builder& output_layer(std::size_t size);
        network_builder& hidden_layer(std::size_t size);
        std::unique_ptr<neural_network> build();

      private:
        std::size_t inputLayer;
        std::size_t outputLayer;
        std::vector<std::size_t> hiddenLayers;
        friend neural_network;
    };

  public:
    /** @note Possible because https://en.cppreference.com/w/cpp/language/lifetime :
     * "Temporary objects are destroyed as the last
     * step in evaluating the full-expression that (lexically)
     * contains the point where they were created."
     */
    static network_builder create();
    neural_network(const self_type&) = delete;
    self_type& operator=(const self_type&) = delete;
    neural_network(self_type&& other) noexcept;
    self_type operator=(self_type&& other) noexcept;
    ~neural_network();

    void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& expected);
    void evaluate(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& expected);

    std::vector<float> predict(const std::vector<float>& input);
    std::vector<std::vector<float>> predict(const std::vector<std::vector<float>>& input);

  private:
    const std::size_t nbLayers;
    std::vector<size_t> nbNeurons;
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<float>> deltas;
    std::vector<std::vector<std::vector<float>>> weights;
    std::vector<std::vector<std::vector<float>>> dWeights;
    std::vector<std::vector<float>> biases;
    std::vector<std::vector<float>> dBiases;
    float learningRate = 0.1f;
    float momentum     = 0.9f;

  private:
    friend class network_builder;
    neural_network(network_builder& nb);

    void initialize_outputs();
    void initialize_weights();
    void initialize_deltas();
    void initialize_biases();

    void forward_pass();
    void backpropagate_error(const std::vector<float>& expected);
    void update_weights();
};

neural_network::network_builder neural_network::create()
{
    return network_builder();
}
inline neural_network::network_builder& neural_network::network_builder::input_layer(std::size_t size)
{
    inputLayer = size;
    return *this;
}
inline neural_network::network_builder& neural_network::network_builder::output_layer(std::size_t size)
{
    outputLayer = size;
    return *this;
}
inline neural_network::network_builder& neural_network::network_builder::hidden_layer(std::size_t size)
{
    hiddenLayers.push_back(size);
    return *this;
}
inline std::unique_ptr<neural_network> neural_network::network_builder::build()
{
    return std::unique_ptr<neural_network>(new neural_network(*this));
}
inline neural_network::neural_network(network_builder& nb):
    nbLayers(nb.hiddenLayers.size() + 2)
{
    // set the number of inputs + bias
    nbNeurons.push_back(nb.inputLayer);
    nbNeurons.insert(nbNeurons.end(), nb.hiddenLayers.begin(), nb.hiddenLayers.end());
    nbNeurons.push_back(nb.outputLayer);

    initialize_outputs();
    initialize_weights();
    initialize_deltas();
    initialize_biases();
}
inline neural_network::~neural_network()
{
}
inline void neural_network::train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& expected)
{
    assert(inputs.front().size() == outputs.front().size());

    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

    for(size_t i = 0; i < inputs.size(); ++i)
    {
        auto now     = std::chrono::high_resolution_clock::now();
        size_t index = indices.back();
        indices.pop_back();
        outputs[0] = inputs[index];
        forward_pass();
        backpropagate_error(expected[index]);
        update_weights();
        std::cout << "[Pass " << i << "/" << inputs.size() << " completed in " << (std::chrono::high_resolution_clock::now() - now).count() * 0.000000001 << " seconds]";
        float MSE = 0.f;
        for(size_t j = 0; j < expected[index].size(); ++j)
        {
            MSE += expected[index][j] - outputs.back()[j];
        }
        auto err = MSE / expected[index].size();
        std::cout << " MSE = " << err << '\n';
    }
}
inline void neural_network::forward_pass()
{
    for(size_t i = 0; i < nbLayers - 1; ++i)
    {
        for(size_t j = 0; j < nbNeurons[i + 1]; ++j)
        {
            float sum = biases[i][j];
            for(size_t k = 0; k < nbNeurons[i]; ++k)
            {
                sum += weights[i][j][k] * outputs[i][k];
            }
            outputs[i + 1][j] = 1.f / (1.f + std::exp(-sum));
        }
    }
}
inline void neural_network::backpropagate_error(const std::vector<float>& expected)
{
    for(size_t i = 0; i < nbNeurons.back(); ++i)
    {
        deltas.back()[i] = (expected[i] - outputs.back()[i]) * outputs.back()[i] * (1.f - outputs.back()[i]);
    }
    for(size_t i = nbLayers - 2; i > 0; --i)
    {
        for(size_t j = 0; j < nbNeurons[i]; ++j)
        {
            float delta = 0.f;
            for(size_t k = 0; j < nbNeurons[i + 1]; ++k)
            {
                delta += weights[i][k][j] * deltas[i][k];
            }
            deltas[i - 1][j] = delta * outputs[i][j] * (1.f - outputs[i][j]);
        }
    }
}
inline void neural_network::update_weights()
{
    for(size_t i = 1; i < nbLayers; ++i)
    {
        for(size_t j = 0; j < nbNeurons[i]; ++j)
        {
            for(size_t k = 0; k < nbNeurons[i - 1]; ++k)
            {
                dWeights[i][j][k] = learningRate * deltas[i][j] * outputs[i - 1][k] + momentum * dWeights[i][j][k];
                weights[i][j][k] += dWeights[i][j][k];
            }
            dBiases[i][j] = deltas[i][j];
            biases[i][j] += dBiases[i][j];
        }
    }
}
inline std::vector<float>
neural_network::predict(const std::vector<float>& input)
{
    outputs[0] = input;
    forward_pass();
    return outputs.back();
}
inline std::vector<std::vector<float>>
neural_network::predict(const std::vector<std::vector<float>>& input)
{
    std::vector<std::vector<float>> out;
    for(size_t i = 0; i < input.size(); ++i)
    {
        out.push_back(predict(input[i]));
    }
    return out;
}
inline void neural_network::initialize_outputs()
{
    std::cout << "Initializing outputs: ";
    outputs.reserve(nbLayers);
    for(const auto& n : nbNeurons)
    {
        outputs.emplace_back(n);
    }
    std::cout << "done\n";
}
inline void neural_network::initialize_weights()
{
    std::cout << "Initializing weights: ";
    weights.reserve(nbLayers - 1);
    dWeights.reserve(nbLayers - 1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.f, 1.f);
    float a = 2.38f;
    for(size_t i = 1; i < nbLayers; ++i)
    {
        float initValue = a / sqrt(nbNeurons[i]);
        weights.emplace_back();
        dWeights.emplace_back();
        for(size_t j = 0; j < nbNeurons[i]; ++j)
        {
            weights.back().emplace_back(nbNeurons[i - 1]);
            dWeights.back().emplace_back(nbNeurons[i - 1]);
            for(size_t k = 0; k < nbNeurons[i - 1]; ++k)
            {
                weights.back().back().push_back(dist(gen) * initValue);
                dWeights.back().back().push_back(0);
            }
        }
    }
    std::cout << "done\n";
}
inline void neural_network::initialize_deltas()
{
    std::cout << "Initializing deltas: ";
    deltas.reserve(nbLayers - 1);
    for(size_t i = 1; i < nbLayers; ++i)
    {
        deltas.emplace_back(nbNeurons[i]);
        std::cout << i << ' ';
    }
    std::cout << "done\n";
}
inline void neural_network::initialize_biases()
{
    std::cout << "Initializing biases: ";
    biases.reserve(nbLayers - 1);
    dBiases.reserve(nbLayers - 1);
    for(size_t i = 1; i < nbLayers; ++i)
    {
        biases.emplace_back(nbNeurons[i], 0);
        dBiases.emplace_back(nbNeurons[i], 0);
    }
    std::cout << "done\n";
}
} // namespace biai
