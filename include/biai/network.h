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

#include <Eigen/Dense>

namespace biai {
// TODO:	implement batch vs mini batch vs stochastic gradient descent
class dataset
{
  public:
    using entry_t = std::pair<Eigen::VectorXf, Eigen::VectorXf>;

  public:
    dataset() = default;
    dataset(std::vector<entry_t>& entries);
    void balance();

    void set_entries(std::vector<entry_t>& entries);

  private:
    std::vector<entry_t> entries;
    std::vector<entry_t> testEntries;
    friend class neural_network;
};
class neural_network
{
  public:
    template<class T>
    using array_t   = std::vector<T>;
    using self_type = neural_network;
    using size_type = std::size_t;
    using matrix_t  = Eigen::MatrixXf;
    using vector_t  = Eigen::VectorXf;
    using rvector_t = Eigen::RowVectorXf;

  public:
    class network_builder
    {
      public:
        network_builder& input_layer(std::size_t size);
        network_builder& output_layer(std::size_t size);
        network_builder& hidden_layer(std::size_t size);
        network_builder& learning_rate(float value);
        network_builder& training_momentum(float value);
        network_builder& learn_stochastic(bool value);
        network_builder& max_epochs(size_t value);
        std::unique_ptr<neural_network> build();

      private:
        std::size_t inputLayer;
        std::size_t outputLayer;
        std::vector<std::size_t> hiddenLayers;
        float learningRate   = 0.1;
        float momentum       = 0.9;
        bool learnStochastic = true;
        size_t maxEpochs     = 10;
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

    void train(const dataset& ds);
    void evaluate(const dataset& ds);

    vector_t predict(const vector_t& input);
    array_t<vector_t> predict(const array_t<vector_t>& input);

  private:
    const std::size_t nbLayers;
    std::vector<size_t> nbNeurons;
    std::vector<vector_t> outputs;
    std::vector<vector_t> deltas;
    std::vector<matrix_t> weights;
    std::vector<matrix_t> dWeights;
    std::vector<vector_t> biases;
    std::vector<vector_t> dBiases;
    float learningRate   = 0.1f;
    float momentum       = 0.9f;
    bool learnStochastic = true;
    size_t maxEpochs;

  private:
    friend class network_builder;
    neural_network(network_builder& nb);

    void initialize_outputs();
    void initialize_weights();
    void initialize_deltas();
    void initialize_biases();

    void train_epoch(const std::vector<dataset::entry_t>& trainingSet);
    float predict_test(const std::vector<dataset::entry_t>& testSet);

    void forward_pass(const vector_t& trainingSet);
    void backpropagate_error(const dataset::entry_t& trainingSet);
    void update_weights();
};
dataset::dataset(std::vector<entry_t>& entries)
{
    std::random_device dev;
    std::mt19937 rng(dev());

    this->entries.assign(entries.begin(), entries.end());
    std::shuffle(this->entries.begin(), this->entries.end(), rng);
}
void dataset::set_entries(std::vector<entry_t>& entries)
{
    std::random_device dev;
    std::mt19937 rng(dev());

    this->entries.assign(entries.begin(), entries.end());
    std::shuffle(this->entries.begin(), this->entries.end(), rng);
}
void dataset::balance()
{
    const size_t dataSize     = entries.size();
    const size_t trainingSize = 0.8 * dataSize;
    auto iterToTest           = std::next(entries.begin(), trainingSize);
    testEntries.insert(testEntries.end(), std::make_move_iterator(iterToTest),
                       std::make_move_iterator(entries.end()));
    entries.erase(iterToTest, entries.end());
}
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
neural_network::network_builder& neural_network::network_builder::learning_rate(float value)
{
    learningRate = value;
    return *this;
}
neural_network::network_builder& neural_network::network_builder::training_momentum(float value)
{
    momentum = value;
    return *this;
}
neural_network::network_builder& neural_network::network_builder::learn_stochastic(bool value)
{
    learnStochastic = value;
    return *this;
}

neural_network::network_builder& neural_network::network_builder::max_epochs(size_t value)
{
    maxEpochs = value;
    return *this;
}
inline std::unique_ptr<neural_network> neural_network::network_builder::build()
{
    return std::unique_ptr<neural_network>(new neural_network(*this));
}
inline neural_network::neural_network(network_builder& nb):
    nbLayers(nb.hiddenLayers.size() + 2), learningRate(nb.learningRate),
    momentum(nb.momentum), learnStochastic(nb.learnStochastic), maxEpochs(nb.maxEpochs)
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
inline void neural_network::train(const dataset& ds)
{
    for(size_t i = 0; i < maxEpochs; ++i)
    {
        auto now = std::chrono::high_resolution_clock::now();
        train_epoch(ds.entries);
        float MSE = predict_test(ds.testEntries);
        std::cout << "[ epoch " << i << " completed in " << (std::chrono::high_resolution_clock::now() - now).count() * 0.000000001 << "with MSE = " << MSE << '\n';
    }
}
inline void neural_network::train_epoch(const std::vector<dataset::entry_t>& trainingSet)
{
    float mse      = 0.f;
    size_t passCnt = 1;
    for(const auto& entry : trainingSet)
    {
        forward_pass(entry.first);
        backpropagate_error(entry);
        if(learnStochastic) update_weights();
        vector_t MSE = outputs.back() - entry.second;
        MSE          = MSE.cwiseProduct(MSE);
        mse += MSE.sum();
        std::cout << "[ pass " << passCnt++ << " of " << trainingSet.size() << " completed ]\n";
    }
    if(!learnStochastic) update_weights();
    mse /= trainingSet.size();
}
inline float neural_network::predict_test(const std::vector<dataset::entry_t>& testSet)
{
    float mse = 0.f;
    for(const auto& entry : testSet)
    {
        forward_pass(entry.first);
        vector_t MSE = outputs.back() - entry.second;
        MSE          = MSE.cwiseProduct(MSE);
        mse += MSE.sum();
    }
    mse /= testSet.size();
    return mse;
}
inline void neural_network::forward_pass(const vector_t& trainingSet)
{
    outputs[0] = weights[0] * trainingSet + biases[0];
    outputs[0] = outputs[0].unaryExpr([this](auto x) { return 1.f / (1.f + std::exp(-x)); });

    for(size_t i = 1; i < nbLayers - 1; ++i)
    {
        outputs[i] = weights[i] * outputs[i - 1] + biases[i];
        outputs[i] = outputs[i].unaryExpr([this](auto x) { return 1.f / (1.f + std::exp(-x)); });
    }
}
inline void neural_network::backpropagate_error(const dataset::entry_t& trainingSet)
{
    auto& outputLayerDeltas  = deltas.back();
    auto& outputLayerOutputs = outputs.back();
    // delta = 2(a - y) o a o (1 - a)
    outputLayerDeltas = (trainingSet.second - outputLayerOutputs);
    outputLayerDeltas = outputLayerDeltas.cwiseProduct(outputLayerOutputs);
    const auto nld    = vector_t::Ones(nbNeurons.back(), 1) - outputLayerOutputs;
    outputLayerDeltas = outputLayerDeltas.cwiseProduct(nld);

    for(size_t i = nbLayers - 2; i > 0; --i)
    {
        // delta = a o (1 - a) o W^T * delta(n + 1)
        auto& layerDeltas     = deltas[i - 1];
        auto& layerWeights    = weights[i];
        auto& nextLayerDeltas = deltas[i];
        auto& layerOutputs    = outputs[i - 1];
        const auto omo        = vector_t::Ones(nbNeurons[i], 1) - layerOutputs;
        layerDeltas           = layerOutputs.cwiseProduct(omo);
        const auto nld        = layerWeights.transpose() * nextLayerDeltas;
        layerDeltas           = layerDeltas.cwiseProduct(nld);
    }
    // input layer
    if(learnStochastic)
    {
        dBiases[0]  = learningRate * deltas[0] + momentum * dBiases[0];
        dWeights[0] = learningRate * deltas[0] * trainingSet.first.transpose() + momentum * dWeights[0];
    }
    else
    {
        dBiases[0] += learningRate * deltas[0];
        dWeights[0] += learningRate * deltas[0] * trainingSet.first.transpose();
    }

    for(size_t i = 1; i < nbLayers - 1; ++i)
    {
        if(learnStochastic)
        {
            dBiases[i]  = learningRate * deltas[i] + momentum * dBiases[i];
            dWeights[i] = learningRate * deltas[i] * outputs[i - 1].transpose() + momentum * dWeights[i];
        }
        else
        {
            dBiases[i] += learningRate * deltas[i];
            dWeights[i] += learningRate * deltas[i] * outputs[i - 1].transpose();
        }
    }
}
inline void neural_network::update_weights()
{
    for(size_t i = 0; i < nbLayers - 1; ++i)
    {
        biases[i] += dBiases[i];
        dWeights[i] += dWeights[i];
    }
    if(!learnStochastic)
    {
        for(size_t i = 0; i < nbLayers - 1; ++i)
        {
            dBiases[i].setZero();
            dWeights[i].setZero();
        }
    }
}
inline neural_network::vector_t
neural_network::predict(const vector_t& input)
{
    forward_pass(input);
    // return outputs.back().unaryExpr([](auto el) { return el < 0.1f ? 0.f : (el > 0.9f ? 1.f : -1.f); });
    return outputs.back();
}
inline neural_network::array_t<neural_network::vector_t>
neural_network::predict(const array_t<vector_t>& input)
{
    array_t<vector_t> out;
    for(size_t i = 0; i < input.size(); ++i)
    {
        out.push_back(predict(input[i]));
    }
    return out;
}
inline void neural_network::initialize_outputs()
{
    std::cout << "Initializing outputs: ";
    outputs.reserve(nbLayers - 1);
    for(size_t i = 1; i < nbLayers; ++i)
    {
        outputs.emplace_back(nbNeurons[i]);
    }
    std::cout << "done\n";
}
inline void neural_network::initialize_weights()
{
    std::cout << "Initializing weights: ";
    weights.reserve(nbLayers - 1);
    dWeights.reserve(nbLayers - 1);
    float a = 2.38f;
    for(size_t i = 1; i < nbLayers; ++i)
    {
        float initValue = a / sqrt(nbNeurons[i - 1]);
        weights.emplace_back(Eigen::MatrixXf::Random(nbNeurons[i], nbNeurons[i - 1]));
        weights.back() = weights.back().unaryExpr([&](auto e) { return e * initValue; });
        dWeights.emplace_back(Eigen::MatrixXf::Zero(nbNeurons[i], nbNeurons[i - 1]));
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
        biases.emplace_back(vector_t::Zero(nbNeurons[i]));
        dBiases.emplace_back(vector_t::Zero(nbNeurons[i]));
    }
    std::cout << "done\n";
}
} // namespace biai
