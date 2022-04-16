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
        network_builder& learning_rate(float value);
        std::unique_ptr<neural_network> build();

      private:
        float learningRate;
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

    void train(const array_t<float>& inputs, const array_t<float>& expected);
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

    size_type inputSize;
    array_t<layer_t> layers;
    activation_function_t af;
    float learningRate;

  private:
    friend class network_builder;
    neural_network(network_builder& nb);

    void initialize_weights();
    array_t<float> forward_propagate(const array_t<float>& inputs);
    void backpropagate_error(const array_t<float>& expected);
    void update_weights(const array_t<float>& inputs);
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
inline neural_network::network_builder& neural_network::network_builder::learning_rate(float value)
{
    learningRate = value;
    return *this;
}
inline std::unique_ptr<neural_network> neural_network::network_builder::build()
{
    return std::unique_ptr<neural_network>(new neural_network(*this));
}
inline neural_network::neural_network(network_builder& nb)
{
    // set the learning rate
    learningRate = nb.learningRate;
    // store the activation function
    af = nb.af;
    // set the number of inputs
    inputSize = nb.inputLayer;
    // create requested amount of hidden layers
    std::for_each(
        nb.hiddenLayers.begin(), nb.hiddenLayers.end(),
        [this](size_type size) {
            layers.push_back(layer_t(size));
        });
    // set the output layer
    layers.push_back(layer_t(nb.outputLayer));

    // set the number of weights in the first hidden layer
    std::for_each(
        layers.front().begin(), layers.front().end(),
        [this](auto& neuron) {
            neuron.weights.resize(inputSize);
        });
    // set the number of weights in the rest of hidden layers and
    // in the output layer
    for(size_type i = 1; i < layers.size(); ++i)
    {
        std::for_each(
            layers[i].begin(), layers[i].end(),
            [&](auto& neuron) {
                neuron.weights.resize(layers[i - 1].size());
            });
    }

    // random number generator
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.f, 1.f);

    // for each weight in output and hidden layers generate a pseudorandom <0,1> number
    // initialize weights to a pseudo-random number between '0' and '1'
    // initialize bias to '0'
    std::for_each(
        layers.begin(), layers.end(),
        [&](auto& layer) {
            std::for_each(
                layer.begin(), layer.end(),
                [&](auto& neuron) {
                    neuron.bias = 0;
                    std::for_each(
                        neuron.weights.begin(), neuron.weights.end(),
                        [&](auto& weight) {
                            weight = distribution(generator);
                        });
                });
        });
}
inline void neural_network::train(const array_t<float>& inputs, const array_t<float>& expected)
{
    assert(inputs.size() == inputSize);
    assert(layers.back().size() == expected.size());
}
inline neural_network::array_t<float> neural_network::forward_propagate(const array_t<float>& inputs)
{
    auto input = inputs;
    for(auto& layer : layers)
    {
        array_t<float> output;
        for(auto& neuron : layer)
        {
            auto& weights = neuron.weights;
            assert(weights.size() == input.size());
            // calculate weighted sum of inputs
            float result = 0.f;
            for(size_type i = 0; i < input.size(); ++i)
            {
                result += weights[i] * input[i];
            }
            // add bias
            result += neuron.bias;
            // add result of activation function to output
            auto activation = af(result);
            neuron.output   = activation;
            output.push_back(activation);
        }
        input = std::move(output);
    }
    return input;
}
inline void neural_network::backpropagate_error(const array_t<float>& expected)
{
    // TODO: calculate with beta
    //
    // backward propagation for the output layer
    auto& outputLayer = layers.back();
    assert(outputLayer.size() == expected.size());
    for(size_type i = 0; i < outputLayer.size(); ++i)
    {
        auto& neuron = outputLayer[i];
        neuron.delta = (neuron.output - expected[i]) * neuron.output * (1.f - neuron.output);
    }
    // backward propagation for the hidden layers
    for(int i = layers.size() - 2; i >= 0; --i)
    {
        auto& layer = layers[i];
        for(size_type j = 0; j < layer.size(); ++j)
        {
            float error = 0.f;
            for(const auto& neuron : layers[i + 1])
            {
                error += neuron.weights[j] * neuron.delta;
            }
            auto& neuron = layer[j];
            neuron.delta = neuron.output * (1.f - neuron.output) * error;
        }
    }
}
inline void neural_network::update_weights(const array_t<float>& inputs)
{
    auto& first = layers.front();
    assert(first.front().weights.size() == inputs.size());

    auto output = inputs;
    // for each layer
    std::for_each(
        layers.begin(), layers.end(),
        [&, this](auto& layer) {
            array_t<float> layerOut;
            // for each neuron
            std::for_each(
                layer.begin(), layer.end,
                [&, this](auto& neuron) {
                    // for each weight
                    for(size_type i = 0; i < neuron.weights.size(); ++i)
                    {
                        // update each weight
                        neuron.weights[i] -= learningRate * neuron.delta * output[i];
                    }
                    // update bias
                    neuron.bias -= learningRate * neuron.delta;
                    layerOut.push_back(neuron.output);
                });
            output = std::move(layerOut);
        });
}
inline neural_network::array_t<float>
neural_network::predict(const array_t<float>& input)
{
    return forward_propagate(input);
}
inline neural_network::array_t<neural_network::array_t<float>>
neural_network::predict(const array_t<array_t<float>>& input)
{
    array_t<array_t<float>> output;
    for(auto& in : input)
    {
        output.push_back(forward_propagate(in));
    }
    return output;
}

} // namespace biai
