#include <iostream>
#include <vector>
#include <random>

class Perceptron {
 public:
  explicit Perceptron(const std::vector<std::vector<float>>& training_data, const std::vector<float>& labels,
					  const size_t& epochs, const float& learning_rate) : bias(1), weights(training_data[0].size()) {
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<float> distribution(-1.0, 1.0);
	for (float& weight : weights) {
	  weight = distribution(generator);
	}
	bias_weight = distribution(generator);

	Train(training_data, labels, epochs, learning_rate);
  }

  float Predict(const std::vector<float>& input) {
	float sum = bias * bias_weight;
	for (size_t i = 0; i < input.size(); ++i) {
	  sum += input[i] * weights[i];
	}
	return (sum > 0) ? 1 : 0;
  }

  void Train(const std::vector<std::vector<float>>& training_data, const std::vector<float>& labels, const size_t& epochs, const float& learning_rate) {
	for (size_t epoch = 0; epoch < epochs; ++epoch) {
	  for (size_t i = 0; i < training_data.size(); ++i) {
		float error = labels[i] - Predict(training_data[i]);

		bias_weight += learning_rate * error * bias;
		for (size_t j = 0; j < weights.size(); ++j) {
		  weights[j] += learning_rate * error * training_data[i][j];
		}
	  }
	}
  }

 private:
  std::vector<float> weights;
  float bias;
  float bias_weight;
};

int main() {
  std::vector<std::vector<float>> training_data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<float> labels = {0, 0, 0, 1};
  Perceptron perceptron(training_data, labels, 1000, 1);
  std::cout << perceptron.Predict({0, 0}) << " " << perceptron.Predict({0, 1}) << " " << perceptron.Predict({1, 0}) << " " << perceptron.Predict({1, 1}) << " " << std::endl;
  return 0;
}
