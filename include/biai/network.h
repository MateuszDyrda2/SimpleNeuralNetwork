#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace biai {
/**  Class representing a dataset containing input and expected output */
class dataset
{
  public:
    /** a single entry pair of input - expected output */
    using entry_t = std::pair<std::vector<float>, std::vector<float>>;

  public:
    /** @brief Default constructor */
    dataset() = default;
    /** @brief Creates a dataset specifying entries
     * @param entries dataset entries
     */
    dataset(std::vector<entry_t>& entries);
    /** @brief Splits the dataset 80/20 on training and validation data */
    void balance();
    /** @brief Attached entries to dataset
     * @param entries dataset entries
     */
    void set_entries(std::vector<entry_t>& entries);

  private:
    std::vector<entry_t> entries;       ///< training set
    std::vector<entry_t> validationSet; ///< validation set
    friend class neural_network;
};
class neural_network
{
  private:
    /** A help type representing a matrix */
    class matrix_t
    {
        std::vector<float> data; ///< internal buffer
        size_t columns;          ///< number of columns

      public:
        /** @brief Constructs a matrix
         * @param rows number of rows
         * @param columns number of columns
         */
        matrix_t(size_t rows, size_t columns):
            data(rows * columns), columns(columns) { }
        /** @brief Function operator used to index the matrix elements
         * @param i row index
         * @param j column index
         * @return reference to the element at the index
         * @warning the indices must be inside the matrix!
         */
        float& operator()(size_t i, size_t j)
        {
            assert(i < data.size() / columns && j < columns);
            return data[i * columns + j];
        }
        /** @brief Function operator used to index the matrix elements
         * @param i row index
         * @param j column index
         * @return value of the element at the index
         * @warning the indices must be inside the matrix!
         */
        float operator()(size_t i, size_t j) const
        {
            assert(i < data.size() / columns && j < columns);
            return data[i * columns + j];
        }
        /** @return number of columns in the matrix */
        size_t get_width() const { return columns; }
        /** @return number of rows in the matrix */
        size_t get_height() const { return data.size() / columns; }
    };

  public:
    /** Available activation functions */
    enum class activation
    {
        Sigmoid,
        ReLU,
        Softmax
    };

  public:
    /** Builder for the neural network */
    class network_builder
    {
      public:
        /** @brief Specify the input layer size
         * @param size number of neurons in the input layer
         * @return reference to the builder
         */
        network_builder& input_layer(std::size_t size);
        /** @brief Specify the output layer size
         * @param size number of neurons in the output layer
         * @return reference to the builder
         */
        network_builder& output_layer(std::size_t size);
        /** @brief Add a hidden layer
         * @param size number of neurons in the hidden layer
         * @param act activation function to be used
         * @return reference to the builder
         */
        network_builder& hidden_layer(std::size_t size, activation act);
        /** @brief Set the learing rate of the network
         * @param value learning rate to set
         * @return reference to the builder
         */
        network_builder& learning_rate(float value);
        /** @brief Set the number of epochs the network will learn for
         * @param value maximum epoch size
         * @return reference to the builder
         */
        network_builder& max_epochs(size_t value);
        /** @brief Create the neural network with the parameters set
         * @return created neural network instance
         */
        std::unique_ptr<neural_network> build();

      private:
        std::size_t inputLayer;
        std::size_t outputLayer;
        std::vector<std::size_t> hiddenLayers;
        std::vector<activation> activations;
        float learningRate = 0.1;
        size_t maxEpochs   = 10;
        friend neural_network;
    };

  public:
    /** @brief Create the neural network instance
     * @return a network builder
     * @note Possible because https://en.cppreference.com/w/cpp/language/lifetime :
     * "Temporary objects are destroyed as the last
     * step in evaluating the full-expression that (lexically)
     * contains the point where they were created."
     */
    static network_builder create();
    neural_network(const neural_network&) = delete;
    neural_network& operator=(const neural_network&) = delete;
    neural_network(neural_network&& other) noexcept;
    neural_network operator=(neural_network&& other) noexcept;
    ~neural_network();
    /** @brief Train the neural network on dataset
     * @param ds dataset to train the neural network on
     */
    void train(const dataset& ds);
    /** @brief Evaluate the trained neural network
     * @param ds dataset to check the neural network against
     */
    void evaluate(const dataset& ds);
    /** @brief Predict the output of the given vector of values
     * @param input values to predict from
     * @return probability of each class
     */
    std::vector<float> predict(const std::vector<float>& input);
    /** @brief Predict the output of the given multiple vectors of values
     * @param input values to predict from
     * @return probability of each class
     */
    std::vector<std::vector<float>> predict(const std::vector<std::vector<float>>& input);

  private:
    const std::size_t nbLayers;
    std::vector<size_t> nbNeurons;
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<float>> deltas;
    std::vector<matrix_t> weights;
    std::vector<matrix_t> dWeights;
    std::vector<std::vector<float>> biases;
    std::vector<std::vector<float>> dBiases;
    std::vector<std::function<float(float)>> activations;
    std::vector<std::function<float(float)>> actDer;
    std::vector<activation> actTypes;

    std::vector<float> cappedOutput;
    std::vector<float> eOutput;
    float learningRate = 0.1f;
    size_t maxEpochs   = 10;

  private:
    friend class network_builder;
    neural_network(network_builder& nb);

    void initialize_outputs();
    void initialize_weights();
    void initialize_deltas();
    void initialize_biases();

    std::pair<float, float> train_epoch(const std::vector<dataset::entry_t>& trainingSet);
    std::pair<float, float> validate_output(const std::vector<dataset::entry_t>& validationSet);

    void forward_pass(const std::vector<float>& trainingSet);
    void backpropagate_error(const dataset::entry_t& trainingSet);
    void update_weights();

    static float sigmoid_activation(float x)
    {
        return 1.f / (1.f + std::exp(-x));
    }
    static float sigmoid_der(float x)
    {
        return x * (1.f - x);
    }
    static float relu_activation(float x)
    {
        return std::max(x, 0.f);
    }
    static float relu_der(float x)
    {
        return x <= 0.f ? 0.f : 1.f;
    }
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

    auto iterToValidate = std::next(entries.begin(), trainingSize);

    validationSet.insert(validationSet.end(), std::make_move_iterator(iterToValidate),
                         std::make_move_iterator(entries.end()));
    entries.erase(iterToValidate, entries.end());
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
inline neural_network::network_builder&
neural_network::network_builder::hidden_layer(std::size_t size, activation act)
{
    hiddenLayers.push_back(size);
    activations.push_back(act);
    return *this;
}
neural_network::network_builder& neural_network::network_builder::learning_rate(float value)
{
    learningRate = value;
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
    maxEpochs(nb.maxEpochs)
{
    // set the number of inputs + bias
    nbNeurons.push_back(nb.inputLayer);
    nbNeurons.insert(nbNeurons.end(), nb.hiddenLayers.begin(), nb.hiddenLayers.end());
    for(size_t i = 0; i < nb.hiddenLayers.size(); ++i)
    {
        switch(nb.activations[i])
        {
        case activation::Sigmoid:
            activations.push_back(&neural_network::sigmoid_activation);
            actDer.push_back(&neural_network::sigmoid_der);
            break;
        case activation::ReLU:
            activations.push_back(&neural_network::relu_activation);
            actDer.push_back(&neural_network::relu_der);
            break;

        default:
            break;
        }
    }
    actTypes = nb.activations;
    actTypes.push_back(activation::Softmax);
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
        auto now          = std::chrono::high_resolution_clock::now();
        auto&& [mse, acc] = train_epoch(ds.entries);
        auto&& [MSE, ACC] = validate_output(ds.validationSet);
        std::cout << "\n[ epoch " << i << " completed in "
                  << (std::chrono::high_resolution_clock::now() - now).count() * 0.000000001
                  << " with MSE = " << MSE << " and accuracy = " << ACC
                  << " epoch MSE = " << mse << " and accuracy = " << acc << " ]\n";
    }
    auto&& [MSE, acc] = validate_output(ds.validationSet);
    std::cout << "\nTest set with MSE = " << MSE << " and accuracy = " << acc << '\n';
}
inline void neural_network::evaluate(const dataset& ds)
{
    auto&& [MSE, ACC] = validate_output(ds.entries);
    std::cout << "\nNetwork evaluted with MSE = " << MSE << " and accuracy = " << ACC << '\n';
}
inline std::pair<float, float>
neural_network::train_epoch(const std::vector<dataset::entry_t>& trainingSet)
{
    float mse      = 0.f;
    size_t errors  = 0;
    float accuracy = 0.f;
    size_t passCnt = 1;
    for(const auto& entry : trainingSet)
    {
        forward_pass(entry.first);
        backpropagate_error(entry);
        update_weights();
        for(size_t i = 0; i < outputs.back().size(); ++i)
        {
            mse += (outputs.back()[i] - entry.second[i]) * (outputs.back()[i] - entry.second[i]);
        }
        std::cout << "\r[ pass " << passCnt++ << " of " << trainingSet.size() << " completed ]" << std::flush;
        if(cappedOutput != entry.second) ++errors;
    }
    mse /= (trainingSet.size() * nbNeurons.back());
    accuracy = 100.f - (errors / float(trainingSet.size())) * 100.f;
    return { mse, accuracy };
}
inline std::pair<float, float>
neural_network::validate_output(const std::vector<dataset::entry_t>& validationSet)
{
    float mse      = 0.f;
    size_t errors  = 0;
    float accuracy = 0.f;
    for(const auto& entry : validationSet)
    {
        forward_pass(entry.first);
        for(size_t i = 0; i < outputs.back().size(); ++i)
        {
            mse += (outputs.back()[i] - entry.second[i]) * (outputs.back()[i] - entry.second[i]);
        }

        if(cappedOutput != entry.second) ++errors;
    }
    mse /= (validationSet.size() * nbNeurons.back());
    accuracy = 100.f - (errors / float(validationSet.size())) * 100.f;
    return { mse, accuracy };
}
inline void neural_network::forward_pass(const std::vector<float>& trainingSet)
{
    for(size_t i = 0; i < nbNeurons[1]; ++i)
    {
        float val = biases[0][i];
        for(size_t j = 0; j < nbNeurons[0]; ++j)
        {
            val += trainingSet[j] * weights[0](i, j);
        }
        outputs[0][i] = activations[0](val);
    }

    for(size_t i = 1; i < nbLayers - 2; ++i)
    {
        for(size_t j = 0; j < nbNeurons[i + 1]; ++j)
        {
            float val = biases[i][j];
            for(size_t k = 0; k < nbNeurons[i]; ++k)
            {
                val += outputs[i - 1][k] * weights[i](j, k);
            }
            outputs[i][j] = activations[i](val);
        }
    }
    // output layer
    for(size_t i = 0; i < nbNeurons[nbLayers - 1]; ++i)
    {
        float val = biases.back()[i];
        for(size_t j = 0; j < nbNeurons[nbLayers - 2]; ++j)
        {
            val += outputs[outputs.size() - 2][j] * weights.back()(i, j);
        }
        eOutput[i] = std::exp(val);
    }
    float sum = std::accumulate(eOutput.begin(), eOutput.end(), 0.f);
    for(size_t i = 0; i < nbNeurons[nbLayers - 1]; ++i)
    {
        outputs.back()[i] = eOutput[i] / sum;
    }

    auto iter  = std::max_element(outputs.back().begin(), outputs.back().end());
    auto index = std::distance(outputs.back().begin(), iter);
    std::fill(cappedOutput.begin(), cappedOutput.end(), 0.f);
    cappedOutput[index] = 1.f;
}
inline void neural_network::backpropagate_error(const dataset::entry_t& trainingSet)
{
    for(size_t i = 0; i < nbNeurons.back(); ++i)
    {
        deltas.back()[i] = outputs.back()[i] - trainingSet.second[i];
    }

    for(int i = nbLayers - 3; i >= 0; --i)
    {
        for(size_t j = 0; j < nbNeurons[i + 1]; ++j)
        {
            float val = 0.f;
            for(size_t k = 0; k < nbNeurons[i + 2]; ++k)
            {
                val += deltas[i + 1][k] * weights[i + 1](k, j);
            }
            deltas[i][j] = actDer[i](outputs[i][j]) * val;
        }
    }
    for(size_t i = 0; i < nbNeurons[1]; ++i)
    {
        dBiases[0][i] = learningRate * deltas[0][i]; // + momentum * dBiases[0][i];
        for(size_t j = 0; j < nbNeurons[0]; ++j)
        {
            dWeights[0](i, j) = learningRate * deltas[0][i] * trainingSet.first[j]; // + momentum * dWeights[0](i, j);
        }
    }
    for(size_t i = 1; i < nbLayers - 1; ++i)
    {
        for(size_t j = 0; j < nbNeurons[i + 1]; ++j)
        {
            dBiases[i][j] = learningRate * deltas[i][j]; // + momentum * dBiases[i][j];
            for(size_t k = 0; k < nbNeurons[i]; ++k)
            {
                dWeights[i](j, k) = learningRate * deltas[i][j] * outputs[i - 1][k]; // + momentum * dWeights[i](j, k);
            }
        }
    }
}
inline void neural_network::update_weights()
{
    for(size_t i = 0; i < nbLayers - 1; ++i)
    {
        for(size_t j = 0; j < nbNeurons[i + 1]; ++j)
        {
            biases[i][j] -= dBiases[i][j];
            for(size_t k = 0; k < nbNeurons[i]; ++k)
            {
                weights[i](j, k) -= dWeights[i](j, k);
            }
        }
    }
}
inline std::vector<float>
neural_network::predict(const std::vector<float>& input)
{
    forward_pass(input);
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
    outputs.reserve(nbLayers - 1);
    for(size_t i = 1; i < nbLayers; ++i)
    {
        outputs.emplace_back(nbNeurons[i]);
    }
    cappedOutput.resize(nbNeurons.back());
    eOutput.resize(nbNeurons.back());
    std::cout << "done\n";
}
inline void neural_network::initialize_weights()
{
    std::cout << "Initializing weights: ";
    weights.reserve(nbLayers - 1);
    dWeights.reserve(nbLayers - 1);
    std::default_random_engine rd;

    for(size_t i = 0; i < nbLayers - 1; ++i)
    {
        float var = 0.f;
        if(actTypes[i] == activation::Sigmoid || actTypes[i] == activation::Softmax)
            var = 2.f / (nbNeurons[i] + nbNeurons[i + 1]); // Xavier
        else
            var = 2.f / (nbNeurons[i]); // He

        std::normal_distribution<float> dist(0.f, var);
        weights.emplace_back(nbNeurons[i + 1], nbNeurons[i]);
        dWeights.emplace_back(nbNeurons[i + 1], nbNeurons[i]);
        for(size_t j = 0; j < nbNeurons[i + 1]; ++j)
        {
            for(size_t k = 0; k < nbNeurons[i]; ++k)
            {
                weights[i](j, k)  = dist(rd);
                dWeights[i](j, k) = 0;
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
