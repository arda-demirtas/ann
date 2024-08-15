#pragma once
#include <iostream>
#include "Layer.h"
#include <vector>
using namespace std;

class Ann
{
private:
	vector<Layer> layers;
public:
	void addLayer(size_t neuron, size_t input_size, string activation_function)
	{
		this->layers.push_back(Layer(neuron, input_size, activation_function));
	}
	MatrixXd predict(MatrixXd& input)
	{
		MatrixXd currentOutput = input;
		for (size_t l = 0; l < this->layers.size(); l++)
		{
			currentOutput = this->layers[l].output(currentOutput);
		}
		return currentOutput;
	}
	double cost(MatrixXd& input, MatrixXd& output)
	{
		MatrixXd sub = this->predict(input) - output;
		MatrixXd pow = sub.cwiseProduct(sub);
		double sum = pow.sum();
		return double(sum) / input.cols();
	}
	void learn(MatrixXd& input, MatrixXd& output, size_t epoch)
	{
		double e = 1e-4;
		double lr = 10;
		vector<Layer> updatedLayers = this->layers;
		for (size_t ep = 0; ep < epoch; ep++)
		{
			double cost = this->cost(input, output);
			cout << "epoch : " << ep << "  cost : " << cost << endl;
			for (size_t l = 0; l < this->layers.size(); l++)
			{
				for (size_t n = 0; n < this->layers[l].weight_matrix.rows(); n++)
				{
					for (size_t w = 0; w < this->layers[l].weight_matrix.cols(); w++)
					{
						double saved = this->layers[l].weight_matrix(n, w);
						this->layers[l].weight_matrix(n, w) += e;
						double dif = (this->cost(input, output) - cost) / e;
						updatedLayers[l].weight_matrix(n, w) = updatedLayers[l].weight_matrix(n, w) - dif * lr;
						this->layers[l].weight_matrix(n, w) = saved;
					}
					
					double saved = this->layers[l].bias_vector(n);
					this->layers[l].bias_vector(n) += e;
					double dif = (this->cost(input, output) - cost) / e;
					updatedLayers[l].bias_vector(n) = updatedLayers[l].bias_vector(n) - dif * lr;
					this->layers[l].bias_vector(n) = saved;
					
				}
			}
			this->layers = updatedLayers;
		}
	}
};