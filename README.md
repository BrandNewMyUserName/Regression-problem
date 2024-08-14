# Neural Network for Function Approximation

This project implements a neural network to approximate the function \( f(x_1, x_2, x_3) = \sin x_1 + \sin x_2 - \cos x_3 \) using backpropagation method. As a result, the network outputs two values. The first is the computed function value, the second one is equal to 1 if the function value is greater than the average value of all points (functions value) or 0 if it's less.


## Neural Network Design

- **Neural Network Structure**: A neural network has 3 input neurons, a one hidden layer with exporing number of hidden neurons, and 2 output neurons.
- **Algorithm**: The backpropagaation method is used.
- **Outputs**:
  - \( exit 1 \): The computed function value.
  - \( exit 2 \): 1 if \( exit 1 \) is greater than the average value, 0 otherwise.


## Function Approximation 

The 100 values of the function for initial argument values and their variations with a step of ±0.5 are calculated.


## Steps

1. **Data Generation**: 
   - Generate input data with variations in the arguments with a step of ±0.5.
   - Compute 100 values of the function and calculate their average.

2. **Data Preparation**: 
   - Normalize each argument relative to all values of its argument in range 0-1.
   - Divide data into training (70% of all data) and test (30%).

3. **Teaching the Network**:
   - Teach the network using the training data.
   - Optimize the network's parameters such as a learning rate, the number of neurons in hidden layer and training accuracy on one epoch to minimize error.
   - Evaluate the speed of the algorithm

4. **Prediction and Evaluation**:
   - Use the trained network to predict the function values for a set of control examples.
   - Choose one of the metrics: MSE, MAE or MAPE to evaluate model's accuracy.
   - Evaluate the speed and accuracy of the algorithm.


## Calculations and taken results




## Conclusion

This project demonstrates how a neural network can be used to approximate mathematical functions, leveraging the backpropagation algorithm for effective learning and prediction.

