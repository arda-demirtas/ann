#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <math.h>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
class Layer
{
private:
	double sigmoid(double data)
	{
		return 1 / (1 + exp(-data));
	}
	double relu(double data)
	{
		if (data >= 0)
		{
			return data;
		}
		else
		{
			return data * 0.001;
		}
	}
	string activation_function;
public:
	MatrixXd weight_matrix;
	VectorXd bias_vector;
	Layer(size_t neuron, size_t input_size, string activation_function)
	{
		this->activation_function = activation_function;
		random_device rd;
		mt19937 gen(rd());
		std::uniform_real_distribution<> distrib_float(-0.1, 0.1);

		this->weight_matrix = MatrixXd(neuron, input_size);
		this->bias_vector = VectorXd(neuron);
		
		for (size_t n = 0; n < neuron; n++)
		{
			for (size_t i = 0; i < input_size; i++)
			{
				this->weight_matrix(n, i) = distrib_float(gen);
			}
			this->bias_vector(n) = distrib_float(gen);
		}

	}
	MatrixXd output(MatrixXd& input)
	{
		MatrixXd z = this->weight_matrix * input;
		MatrixXd z_with_bias = z.colwise() + this->bias_vector;
		
		for (size_t row = 0; row < z_with_bias.rows(); row++)
		{
			for (size_t col = 0; col < z_with_bias.cols(); col++)
			{
				if (this->activation_function == "sigmoid")
				{
					z_with_bias(row, col) = this->sigmoid(z_with_bias(row, col));
				}
				else if (this->activation_function == "relu")
				{
					z_with_bias(row, col) = this->relu(z_with_bias(row, col));
				}
			}
		}
		
		return z_with_bias;
	}
};
