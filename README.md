# Neural Network for Function Approximation

This project implements a manually constructed neural network to approximate the function \( f(x_1, x_2, x_3) = \x_1 * \x_2 - \x_3 \) using backpropagation method. The network outputs two values. The first is the computed function value, the second one is equal to 1 if the function value is greater than the average value of all points (functions value) or 0 if it's less.

## The purpose of Task 
The purpose of the task is to learn how to use neural networks for complex regression problems on the example of a linear function as well as evaluate dependecy between training speed with accuracy of a model and size of training/test sets with the number of neurons in a hidden layer. 

## Used Libraries
- Numpy: for performing fast calculations and effective neural network construction
- Matplotlib: for training speed and error changing visualisation  

## Neural Network Design

- **Neural Network Structure**: A neural network has 3 input neurons, one hidden layer with exporing number of hidden neurons, and 2 output neurons.
- **Algorithm**: The backpropagaation method is used.
- **Outputs**:
  - \( exit 1 \): The computed function value.
  - \( exit 2 \): 1 if \( exit 1 \) is greater than the average value, 0 otherwise.

## Approximate Function  
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

## Analyze of results
For each set of parameters 3 experiments were made. It was done to exclude accidential deviation and to see the whole picture of neural network's perfomance. For the test dataset evaluation MAPE were used. 

**Network parameters:**

Constant: 
- Learning rate: 0.01
- The maximum possible number of epoches: 5000
- The training error on an epoch after which algorithm stops: 0.01

Changeable: 
- The distribution of training/test sets: 50%/50%, 70%/30%, 80%/20%
- The number of neurons in a hidden layer: 10, 20, 30


<details>
  <summary>Click to expand/collapse examples of experimental results</summary>

### Example 1

Training the neural network:
Training sample size: 80 | Number of neurons in the hidden layer: 10 | Learning rate: 0.01
200 iteration: Error 0.05158
400 iteration: Error 0.05036
600 iteration: Error 0.04890
800 iteration: Error 0.04693
1000 iteration: Error 0.04411
1200 iteration: Error 0.04009
1400 iteration: Error 0.03486
1600 iteration: Error 0.02927
1800 iteration: Error 0.02438
2000 iteration: Error 0.02063
2200 iteration: Error 0.01786
2400 iteration: Error 0.01578
2600 iteration: Error 0.01415
2800 iteration: Error 0.01283
3000 iteration: Error 0.01172
3200 iteration: Error 0.01077
Total number of iterations: 3386 | Error in the last iteration: 0.01000 | Metric: MSE


Testing:
Test sample size: 20
========================================================================================================
  X1		    X2		 X3		 y1_pred	 y2_pred	 y1_real	 y2_real	
========================================================================================================
  0.48214	 0.50000	 0.14815	 0.47512	 0.0		 0.38922	 0.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 0.40741	 0.68973	 1.0		 0.64848	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.03704	 0.50000	 0.68827	 1.0		 0.51786	 0.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 0.03704	 0.36968	 0.0		 0.27811	 0.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 0.50000	 0.74595	 1.0		 0.74107	 1.0
--------------------------------------------------------------------------------------------------------
  0.28571	 0.50000	 0.50000	 0.66545	 1.0		 0.64286	 0.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 0.00000	 0.33542	 0.0		 0.24107	 0.0
--------------------------------------------------------------------------------------------------------
  1.00000	 0.50000	 0.50000	 0.87092	 1.0		 1.00000	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 0.18519	 0.50975	 0.0		 0.42626	 0.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.48148	 0.50000	 0.74390	 1.0		 0.73214	 1.0
--------------------------------------------------------------------------------------------------------
  0.57143	 0.50000	 0.50000	 0.77563	 1.0		 0.78571	 1.0
--------------------------------------------------------------------------------------------------------
  0.10714	 0.50000	 0.11111	 0.22570	 0.0		 0.16468	 0.0
--------------------------------------------------------------------------------------------------------
  0.07143	 0.50000	 0.50000	 0.55378	 0.0		 0.53571	 0.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 1.00000	 0.90058	 1.0		 1.24107	 1.0
--------------------------------------------------------------------------------------------------------
  0.89286	 0.50000	 0.50000	 0.85309	 1.0		 0.94643	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.74074	 0.50000	 0.77073	 1.0		 0.85714	 1.0
--------------------------------------------------------------------------------------------------------
  0.32143	 0.50000	 0.50000	 0.68170	 1.0		 0.66071	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 0.81481	 0.86480	 1.0		 1.05589	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.51852	 0.50000	 0.74798	 1.0		 0.75000	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 0.07407	 0.40467	 0.0		 0.31515	 0.0
--------------------------------------------------------------------------------------------------------
Error: 0.15535075002871776 | Metric: MAPE

![Training error changing over iteration](Pictures/Training_error_over_iteration(80-20-dataset 10neurons).png)

### Example 2

Training the neural network:
Training sample size: 80 | Number of neurons in the hidden layer: 30 | Learning rate: 0.01
200 iteration: Error 0.15366
400 iteration: Error 0.15365
600 iteration: Error 0.15365
800 iteration: Error 0.15364
1000 iteration: Error 0.15364
1200 iteration: Error 0.15363
1400 iteration: Error 0.15362
1600 iteration: Error 0.15360
1800 iteration: Error 0.15358
2000 iteration: Error 0.15356
2200 iteration: Error 0.15353
2400 iteration: Error 0.15349
2600 iteration: Error 0.15341
2800 iteration: Error 0.15328
3000 iteration: Error 0.15293
3200 iteration: Error 0.15044
3400 iteration: Error 0.05202
3600 iteration: Error 0.04930
3800 iteration: Error 0.04584
4000 iteration: Error 0.04129
4200 iteration: Error 0.03557
4400 iteration: Error 0.02934
4600 iteration: Error 0.02363
4800 iteration: Error 0.01912
5000 iteration: Error 0.01577
Total number of iterations: 5000 | Error in the last iteration: 0.01577 | Metric: MSE


Testing:
Test sample size: 20
========================================================================================================
  X1		 X2		 X3		 y1_pred	 y2_pred	 y1_real	 y2_real	
========================================================================================================
  0.48214	 0.77778	 0.50000	 0.70777	 1.0		 0.87500	 1.0
--------------------------------------------------------------------------------------------------------
  0.21429	 0.50000	 0.22222	 0.42649	 0.0		 0.32937	 0.0
--------------------------------------------------------------------------------------------------------
  0.28571	 0.50000	 0.50000	 0.67432	 0.0		 0.64286	 0.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 0.70370	 0.82810	 1.0		 0.94478	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 0.25926	 0.55374	 0.0		 0.50033	 0.0
--------------------------------------------------------------------------------------------------------
  0.57143	 0.50000	 0.50000	 0.74825	 1.0		 0.78571	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.33333	 0.50000	 0.73806	 1.0		 0.66071	 0.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 0.18519	 0.49216	 0.0		 0.42626	 0.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.81481	 0.50000	 0.70509	 1.0		 0.89286	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 0.85185	 0.87739	 1.0		 1.09292	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 1.00000	 0.50000	 0.69142	 1.0		 0.98214	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.44444	 0.50000	 0.73084	 1.0		 0.71429	 1.0
--------------------------------------------------------------------------------------------------------
  0.32143	 0.50000	 0.50000	 0.68458	 1.0		 0.66071	 0.0
--------------------------------------------------------------------------------------------------------
  0.03571	 0.50000	 0.50000	 0.59497	 0.0		 0.51786	 0.0
--------------------------------------------------------------------------------------------------------
  0.03571	 0.50000	 0.03704	 0.22659	 0.0		 0.05489	 0.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.96296	 0.50000	 0.69419	 1.0		 0.96429	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.50000	 0.62963	 0.79628	 1.0		 0.87070	 1.0
--------------------------------------------------------------------------------------------------------
  0.96429	 0.50000	 0.50000	 0.82138	 1.0		 0.98214	 1.0
--------------------------------------------------------------------------------------------------------
  0.48214	 0.22222	 0.50000	 0.74500	 1.0		 0.60714	 0.0
--------------------------------------------------------------------------------------------------------
  0.10714	 0.50000	 0.11111	 0.29934	 0.0		 0.16468	 0.0
--------------------------------------------------------------------------------------------------------
Error: 0.3348656455854145 | Metric: MAPE

![Training error changing over iteration](Pictures/Training_error_over_iteration(80-20-dataset 30neurons).png)

</details>

### Detailed statistics

| Train Set Size | Hidden Neurons | Iteration Speed | MAPE Error |
|----------------|----------------|-----------------|------------|
| **50**         | **10**         | 5000            | 529.79     |
|                |                | 5000            | 256.00     |
|                |                | 5000            | 446.25     |
| **50**         | **20**         | 3800            | 0.11       |
|                |                | 3272            | 0.17       |
|                |                | 5000            | 0.20       |
| **50**         | **30**         | 5000            | 0.72       |
|                |                | 5000            | 1900.00    |
|                |                | 5000            | 1247.00    |
| **70**         | **10**         | 4288            | 0.084      |
|                |                | 3022            | 0.205      |
|                |                | 3678            | 0.135      |
| **70**         | **20**         | 2600            | 0.200      |
|                |                | 3022            | 0.205      |
|                |                | 2750            | 0.168      |
| **70**         | **30**         | 5000            | 0.82       |
|                |                | 4915            | 0.132      |
|                |                | 4164            | 0.112      |
| **80**         | **10**         | 3834            | 0.124      |
|                |                | 3386            | 0.155      |
|                |                | 2884            | 0.127      |
| **80**         | **20**         | 2363            | 0.101      |
|                |                | 2274            | 0.090      |
|                |                | 1923            | 0.121      |
| **80**         | **30**         | 5000            | 0.334      |
|                |                | 5000            | 0.756      |
|                |                | 5000            | 0.863      |
|----------------|----------------|-----------------|------------|


Through all experiments, the best perfomance were shown by neural network with 80/20 data distribution and 20 neurons in a hidden layer. The average speed and accuracy are 2186 iterations and 0.10 loss respectively. The worst results were provided by neural network with 50/50 data and 10 neurons. An average accuracy and speed are 5000 iterations and 0.10 loss. The taken results come from correct distribution of data in training and test parts and enough network's power to proccess it in the first case and via versa not enough size of training dataset and small power of network in the second.

## Conclusion

This project demonstrates how a neural network can be used to approximate mathematical functions, leveraging the backpropagation algorithm for effective learning and prediction. Moreover, the taken results empasize the importance of appropriate parameters, in this case sizes of training and test data as well as the number of hidden neurons to get a correct model.

