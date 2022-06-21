# SimpleNeuralNetwork
Simple neural network implemented in c++

## Getting the library
---
Get the code from github   
``` $ git clone https://github.com/MateuszDyrda2/SimpleNeuralNetwork.git ```  
If you want to run the example you need to also get libpng which is a dependancy of the example project:  
``` $ git submodule init && git submodule update ```  

## Running the code 
---
### CMake
To build the project using CMake go into the directory with cloned project and create a build directory:  
``` $ mkdir build && cd build```  
Run CMake inside the build directory:  
``` $ cmake .. ```  
It will generate a makefile on Linux and Visual Studio project on Windows.
## Using the library
---
First create a dataset:
``` C++
vector<dataset::entry_t> data;
data.push_back({ 0.0f, 0.0f }, { 0.0f });
data.push_back({ 1.0f, 0.0f }, { 1.0f });
data.push_back({ 0.0f, 1.0f }, { 1.0f });
data.push_back({ 1.0f, 1.0f }, { 0.0f });
dataset ds(data);
```
Then initialize the neural network with custom parameters:
``` C++
auto nn = neural_network::create()
				.input_layer(2)
				.hidden_layer(2, neural_network::activation::Sigmoid)
				.output_layer(1)
				.learning_rate(0.1)
				.build();
```
Then you can train the neural network:
``` C++
nn->train(ds);
```
After that you can use the neural network to predict values out of your input:
``` C++
auto res = nn->predict({ 1.0f, 0.0f });
```