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

namespace biai {
// TODO: Implement as matrices
static float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}
class neural_network
{
  public:
    template<class T>
    using array_t               = std::vector<T>;
    using self_type             = neural_network;
    using size_type             = std::size_t;
    using activation_function_t = std::function<float(float)>;

  public:
    class network_builder
    {
      public:
        network_builder& input_layer(std::size_t size);
        network_builder& output_layer(std::size_t size);
        network_builder& hidden_layer(std::size_t size);
        network_builder& activation_function(const activation_function_t& af);
        std::unique_ptr<neural_network> build();

      private:
        std::size_t inputLayer;
        std::size_t outputLayer;
        std::vector<std::size_t> hiddenLayers;
        activation_function_t af;
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

    void train(const array_t<float>& inputs, const array_t<float>& expected, float learningRate);
    void evaluate(const array_t<float>& inputs, const array_t<float>& expected);

    array_t<float> predict(const array_t<float>& input);
    array_t<array_t<float>> predict(const array_t<array_t<float>>& input);

  private:
    struct neuron_t
    {
        array_t<float> weights;
        float bias;
        float output;
        float delta;
    };
    using layer_t = array_t<neuron_t>;

    array_t<float> inputLayer;
    array_t<size_type> layerSize;
    array_t<array_t<float>> biases;
    array_t<array_t<float>> outputs;
    array_t<array_t<float>> deltas;
    array_t<array_t<array_t<float>>> weights;

    // array_t<layer_t> layers;
    activation_function_t af;

  private:
    friend class network_builder;
    neural_network(network_builder& nb);

    void initialize_weights();
    array_t<float> forward_pass();
    void backpropagate_error(const array_t<float>& expected);
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
inline neural_network::network_builder& neural_network::network_builder::activation_function(const activation_function_t& af)
{
    this->af = af;
    return *this;
}
inline std::unique_ptr<neural_network> neural_network::network_builder::build()
{
    return std::unique_ptr<neural_network>(new neural_network(*this));
}
inline neural_network::neural_network(network_builder& nb)
{
    // store the activation function
    af = nb.af;
    // set the number of inputs
    inputLayer.resize(nb.inputLayer);
    // set the number of layers
    layerSize.assign(nb.hiddenLayers.begin(), nb.hiddenLayers.end());
    layerSize.push_back(nb.outputLayer);

    for(auto& size : layerSize)
    {
        biases.emplace_back(size);
        outputs.emplace_back(size);
        deltas.emplace_back(size);
        weights.emplace_back(size);
    }

    // set the number of weights in the hidden layers and output layer
    size_type weightSize = inputLayer.size();
    for(auto& wi : weights)
    {
        for(auto& wj : wi)
        {
            wj.resize(weightSize);
        }
        weightSize = wi.size();
    }

    // random number generator
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.f, 1.f);

    // for each weight in output and hidden layers generate a pseudorandom <0,1> number
    // initialize weights to a pseudo-random number between '0' and '1'
    // initialize bias to '0'
    for(auto& wi : weights)
    {
        for(auto& wj : wi)
        {
            for(auto& wk : wj)
            {
                wk = distribution(generator);
            }
        }
    }
}
inline void neural_network::train(const array_t<float>& inputs, const array_t<float>& expected, float learningRate)
{
    assert(inputs.size() == inputLayer.size());
    assert(layerSize.back() == expected.size());
    inputLayer = inputs;
}
inline neural_network::array_t<float> neural_network::forward_pass()
{
    auto* in = &inputLayer;
    for(size_type i = 0; i < layerSize.size(); ++i)
    {
        for(size_type j = 0; j < layerSize[i]; ++j)
        {
            auto& w      = weights[i][j];
            float result = biases[i][j];
            for(size_type k = 0; k < w.size(); ++k)
            {
                result += w[k] * (*in)[k];
            }
            outputs[i][j] = af(result);
        }
        in = &outputs[i];
    }
    return *in;
}
inline void neural_network::backpropagate_error(const array_t<float>& expected)
{
    // TODO: calculate with beta
    //
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
}
inline void neural_network::update_weights(float learningRate)
{
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
}
inline neural_network::array_t<float>
neural_network::predict(const array_t<float>& input)
{
    inputLayer = input;
    return forward_propagate();
}
inline neural_network::array_t<neural_network::array_t<float>>
neural_network::predict(const array_t<array_t<float>>& input)
{
    array_t<array_t<float>> output;
    for(auto& in : input)
    {
        inputLayer = in;
        output.push_back(forward_pass());
    }
    return output;
}

} // namespace biai
