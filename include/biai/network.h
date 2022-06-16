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

struct activation_func
{
    virtual float activate(float x) = 0;
    virtual float gradient(float x) = 0;
};
struct sigmoid : public activation_func
{
    inline float activate(float x) override { return 1.f / (1.f + std::exp(-x)); }
    inline float gradient(float x) override { return x * (1.f - x); }
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
        network_builder& activation_function(std::unique_ptr<activation_func>&& af);
        std::unique_ptr<neural_network> build();

      private:
        std::size_t inputLayer;
        std::size_t outputLayer;
        std::vector<std::size_t> hiddenLayers;
        std::unique_ptr<activation_func> af;
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

    void train(const array_t<vector_t>& inputs, const array_t<vector_t>& expected);
    void evaluate(const array_t<vector_t>& inputs, const array_t<vector_t>& expected);

    vector_t predict(const vector_t& input);
    array_t<vector_t> predict(const array_t<vector_t>& input);

  private:
    const std::size_t nbLayers;
    std::vector<vector_t> outputs;
    std::vector<rvector_t> deltas;
    std::vector<matrix_t> weights;
    std::vector<vector_t> biases;
    float learningRate = 0.1f;

    std::unique_ptr<activation_func> af;

  private:
    friend class network_builder;
    neural_network(network_builder& nb);

    void initialize_weights();
    void forward_pass();
    void backpropagate_error(const vector_t& expected);
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
inline neural_network::network_builder& neural_network::network_builder::activation_function(std::unique_ptr<activation_func>&& af)
{
    this->af = std::move(af);
    return *this;
}
inline std::unique_ptr<neural_network> neural_network::network_builder::build()
{
    return std::unique_ptr<neural_network>(new neural_network(*this));
}
inline neural_network::neural_network(network_builder& nb):
    nbLayers(nb.hiddenLayers.size() + 2)
{
    // store the activation function
    af = std::move(nb.af);
    // set the number of inputs + bias
    auto currentNeuronSize = nb.inputLayer;
    outputs.emplace_back(currentNeuronSize);
    std::cout << "[Created input layer]\n";
    for(const auto& nbNeurons : nb.hiddenLayers)
    {
        outputs.emplace_back(nbNeurons);
        deltas.emplace_back(nbNeurons);
        weights.emplace_back(matrix_t::Random(nbNeurons, currentNeuronSize));
        biases.emplace_back(nbNeurons);
        currentNeuronSize = nbNeurons;
        std::cout << "[Created hidden layer]\n";
    }
    outputs.emplace_back(nb.outputLayer);
    deltas.emplace_back(nb.outputLayer);
    weights.emplace_back(matrix_t::Random(nb.outputLayer, currentNeuronSize));
    biases.emplace_back(nb.outputLayer);
    std::cout << "[Created output layer]\n";

    // for each weight in output and hidden layers generate a pseudorandom <0,1> number
    // initialize weights to a pseudo-random number between '0' and '1'
    // initialize bias to '0'
    // TODO: Initialize to a right value
}
inline neural_network::~neural_network()
{
}
inline void neural_network::train(const array_t<vector_t>& inputs, const array_t<vector_t>& expected)
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
        std::cout << "[Pass " << i << "/" << inputs.size() << " completed in]" << (std::chrono::high_resolution_clock::now() - now).count() * 0.000001 << " seconds\n";
    }
}
inline void neural_network::forward_pass()
{
    for(size_t i = 0; i < nbLayers - 1; ++i)
    {
        outputs[i + 1] = weights[i] * outputs[i] + biases[i];
        outputs[i + 1].unaryExpr([this](auto x) { return af->activate(x); });
    }
}
inline void neural_network::backpropagate_error(const vector_t& expected)
{
    // backward propagation for the output layer
    // delta = (out - exp)
    // delta = (sum(deltas(n+1) * weights(n+1)))
    vector_t delta = outputs.back() - expected;
    delta.binaryExpr(outputs.back(), [this](auto a, auto b) { return a * af->gradient(b); });
    deltas.front() = delta.transpose();

    for(int i = deltas.size() - 2; i >= 0; --i)
    {
        deltas[i] = deltas[i + 1] * weights[i + 1];
        deltas[i].binaryExpr(outputs[i + 1].transpose(), [this](auto a, auto b) { return a * af->gradient(b); });
    }
}
inline void neural_network::update_weights()
{
    for(size_t i = 0; i < nbLayers - 1; ++i)
    {
        weights[i] -= (learningRate * outputs[i] * deltas[i]).transpose();
        biases[i] -= (learningRate * deltas[i]).transpose();
    }
}
inline neural_network::vector_t
neural_network::predict(const vector_t& input)
{
    outputs[0] = input;
    forward_pass();
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

} // namespace biai
