#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
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

    void train(const array_t<vector_t>& inputs, const array_t<vector_t>& expected, float learningRate);
    void evaluate(const array_t<vector_t>& inputs, const array_t<vector_t>& expected);

    vector_t predict(const vector_t& input);
    array_t<vector_t> predict(const array_t<vector_t>& input);

  private:
#if 0
    array_t<float> inputLayer;
    array_t<size_type> layerSize;
    array_t<array_t<float>> biases;
    array_t<array_t<float>> outputs;
    array_t<array_t<float>> deltas;
    array_t<array_t<array_t<float>>> weights;
#endif
    struct layer_t
    {
        vector_t biases;
        vector_t outputs;
        vector_t deltas;
        matrix_t weights;
        layer_t() = default;
        layer_t(size_type prev, size_type size):
            biases(size),
            outputs(size),
            deltas(size),
            weights(matrix_t::Random(prev, size)) { }
    };

    vector_t inputLayer;
    array_t<layer_t> layers;

    std::unique_ptr<activation_func> af;

  private:
    friend class network_builder;
    neural_network(network_builder& nb);

    void initialize_weights();
    vector_t forward_pass();
    void backpropagate_error(const vector_t& expected);
    void update_weights(float learningRate);
};

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
inline neural_network::neural_network(network_builder& nb)
{
    // store the activation function
    af = std::move(nb.af);
    // set the number of inputs
    inputLayer = matrix_t::Constant(nb.inputLayer, 1, 0.f);
    // set the number of layers
    auto prevSize = nb.inputLayer;
    for(auto& size : nb.hiddenLayers)
    {
        layers.emplace_back(prevSize, size);
        prevSize = size;
    }
    layers.emplace_back(prevSize, nb.outputLayer);

    // for each weight in output and hidden layers generate a pseudorandom <0,1> number
    // initialize weights to a pseudo-random number between '0' and '1'
    // initialize bias to '0'
    // TODO: Initialize to a right value
}
inline void neural_network::train(const array_t<vector_t>& inputs, const array_t<vector_t>& expected, float learningRate)
{
    assert(inputs.size() == inputLayer.size());
    for(size_type i = 0; i < inputs.size(); ++i)
    {
        inputLayer = inputs[i];
        forward_pass();
        backpropagate_error(expected[i]);
        update_weights(learningRate);
    }
}
inline neural_network::vector_t neural_network::forward_pass()
{
    auto* in = &inputLayer;
    for(auto& layer : layers)
    {
        layer.outputs = layer.weights * (*in) + layer.biases;
        layer.outputs.unaryExpr([this](auto x) { return af->activate(x); });
        in = &layer.outputs;
    }
    return *in;
}
inline void neural_network::backpropagate_error(const vector_t& expected)
{
    // TODO: calculate with beta
    //
#if 0
    // backward propagation for the output layer
    auto outSize = layerSize.back();
    assert(outSize == expected.size());
    for(size_t i = 0; i < outSize; ++i)
    {
        auto nOutput       = outputs[outSize][i];
        deltas[outSize][i] = (nOutput - expected[i]) * nOutput * (1.f - nOutput);
    }
    // backward propagation for the hidden layers
    for(size_t i = layerSize.size() - 2; i >= 0; --i)
    {
        for(size_t j = 0; j < layerSize[i]; ++j)
        {
            float error = 0.f;
            for(size_t k = 0; k < layerSize[i + 1]; ++k)
            {
                error += weights[i + 1][k][j] * deltas[i + 1][k];
            }
            auto nOutput = outputs[i][j];
            deltas[i][j] = nOutput * (1.f - nOutput) * error;
        }
    }
#endif

    auto& out = layers.back();
    // delta = (out - exp)
    out.deltas = out.outputs - expected;
    out.deltas.binaryExpr(out.outputs, [this](auto a, auto b) { return a * af->gradient(b); });
    // delta = (sum(deltasn+1 * weightsn+1))
    for(size_type i = layers.size() - 2; i >= 0; --i)
    {
        auto& layer  = layers[i];
        auto& next   = layers[i + 1];
        layer.deltas = next.weights * next.deltas;
        layer.deltas.binaryExpr(layer.outputs, [this](auto a, auto b) { return a * af->gradient(b); });
    }
}
inline void neural_network::update_weights(float learningRate)
{
#if 0
    auto in = &inputLayer;
    // for each layer
    for(size_type i = 0; i < weights.size(); ++i)
    {
        // for each neuron
        for(size_type j = 0; j < weights[i].size(); ++j)
        {
            // for each weight
            for(size_type k = 0; k < weights[i][j].size(); ++k)
            {
                weights[i][j][k] -= learningRate * deltas[i][j] * (*in)[k];
            }
            biases[i][j] -= learningRate * deltas[i][j];
        }
        in = &outputs[i];
    }
#endif
    rvector_t in = inputLayer.transpose();
    for(auto& layer : layers)
    {
        auto deltas = learningRate * layer.deltas * in;
        layer.weights -= deltas;
        auto biasDeltas = learningRate * layer.deltas;
        layer.deltas -= biasDeltas;
        in = layer.outputs.transpose();
    }
}
inline neural_network::vector_t
neural_network::predict(const vector_t& input)
{
    inputLayer = input;
    return forward_pass();
}
inline neural_network::array_t<neural_network::vector_t>
neural_network::predict(const array_t<vector_t>& input)
{
    array_t<vector_t> output;
    for(auto& in : input)
    {
        inputLayer = in;
        output.push_back(forward_pass());
    }
    return output;
}

} // namespace biai
