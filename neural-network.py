import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=5, suppress=True)

# Define the target function to approximate
def function(x1, x2, x3) -> np.dtype('float64'): 
    return x1 * x2 + x3 

# Sigmoid activation function
def sigmoid(x) -> np.dtype('float64'):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function for backpropagation
def d_sigmoid(x) -> np.dtype('float64'):
    return x * (1 - x)

# Function to normalize the input data
def normalization(X):
    for i in range(X.shape[1]):  
        x_max = np.max(X[:, i])
        x_min = np.min(X[:, i])

        # Normalize each element
        for j in range(X.shape[0]):  
            X[j, i] = (X[j, i] - x_min) / (x_max - x_min)

    return X

# Evaluate different error metrics
def eval_metrics(y_pred, y_real, metrics):
    if metrics == 'MSE':
        result = (y_pred - y_real)**2  # Mean Squared Error
    elif metrics == 'MAE':
        result = np.abs(y_pred - y_real)  # Mean Absolute Error
    elif metrics == 'MAPE':
        result = np.abs(y_pred - y_real) / (y_real + 1e-5)  # Mean Absolute Percentage Error
    else:
        # Default is MSE
        result = (y_pred - y_real)**2

    return result

# Split the data into training and validation sets
def train_validation_split(X_values, train_size, to_normalize=False):
    np.random.shuffle(X_values)  # Shuffle data to ensure randomness
    
    if to_normalize:
        X_values = normalization(X_values)

    # Compute the function values for the dataset
    Y_values = np.array([function(x, y, z) for x, y, z in X_values])

    # Split the data into training and validation sets
    X_train = X_values[:train_size]
    X_valid = X_values[train_size:]

    Y_train = Y_values[:train_size]
    Y_valid = Y_values[train_size:]

    return X_train, X_valid, Y_train, Y_valid

# Perform a forward pass through the network
def forward_move(W_1, W_2, X, average):
    # Compute hidden layer neuron activations
    activation_1 = np.array(np.dot(W_1, X))
    output_l1 = np.array(sigmoid(activation_1))

    # Compute output layer neuron activations
    activation_2 = np.array(np.dot(output_l1, W_2))
    output_l2 = np.array(sigmoid(activation_2))

    # Determine the network's final output
    output_net = np.array([output_l2[0], 1 if output_l2[0] > average else 0])

    return output_l1, output_l2, output_net 

# Train the neural network using backpropagation
def backpropagation(X, Y, hidden_neurons, lr_rate, threshold=0.15, total_epochs=2000, metric="MSE", show_progress=True):
    print("Training the neural network:")
    print(f"Training sample size: {len(X)} | Number of neurons in the hidden layer: {hidden_neurons} | Learning rate: {lr_rate}")
    
    # Initialize random weights for the layers
    weights_l1 = np.random.rand(hidden_neurons, 3)
    weights_l2 = np.random.rand(hidden_neurons, 2)
    num_of_image = list(range(0, len(X)))
    average = np.mean(Y)  # Calculate the average target value
    errors_array = []
      
    error = 1
    epoch = 0

    while epoch < total_epochs and threshold < error:
        epoch += 1
        error = 0
        np.random.shuffle(num_of_image)  # Shuffle the training data

        for i in range(len(X)):
            x = X[num_of_image[i]]
            y_real = [Y[num_of_image[i]], 1 if Y[num_of_image[i]] > average else 0]

            #### Forward Pass ####
            output_l1, output_l2, y_pred = forward_move(weights_l1, weights_l2, x, average)

            #### Backward Pass ####

            # Adjust weights of the 2nd layer
            delta_2 = (y_pred - y_real) * d_sigmoid(output_l2) 
            weights_l2 += -lr_rate * np.outer(output_l1, delta_2)

            # Adjust weights of the 1st layer
            delta_1 = np.dot(weights_l2, delta_2) * d_sigmoid(output_l1)
            weights_l1 += -lr_rate * np.outer(delta_1, x)

            # Accumulate the error for the batch
            error += eval_metrics(y_pred[0], y_real[0], metric)

        # Calculate average error
        error /= len(X)
        errors_array.append(error)
        
        # Print progress every 200 epochs if enabled
        if epoch % 200 == 0 and show_progress:
            print(f"{epoch} iteration: Error {error:.5f}")  

        if error < threshold:
            break        

    print(f"Total number of iterations: {epoch} | Error in the last iteration: {error:.5f} | Metric: {metric}\n\n")  
    return weights_l1, weights_l2, errors_array

# Test the trained model on validation data
def test_model(X, Y, weights_l1, weights_l2, output=True, metrics="MAE"):
    y_pred_ar = []
    if output:
        print("Testing:")
        print(f"Test sample size: {len(X)}")
        print("========================================================================================================")
        print("  X1\t\t X2\t\t X3\t\t y1_pred\t y2_pred\t y1_real\t y2_real\t")
        print("========================================================================================================")

    total_error = 0
    average = np.mean(Y)

    for k in range(len(X)):

        output_l1, output_l2, y_pred = forward_move(weights_l1, weights_l2, X[k], average)

        y_real = np.array([Y[k], 1 if Y[k] > average else 0])

        # Compute the error for the current sample
        error = eval_metrics(y_pred[0], y_real[0], metrics)
        y_pred_ar.append(y_pred[0])
        total_error += error

        if output:
            print(f"  {'%.5f'%X[k][0]}\t {'%.5f'%X[k][1]}\t {'%.5f'%X[k][2]}\t {'%.5f'%y_pred[0]}\t {y_pred[1]}\t\t {'%.5f'%y_real[0]}\t {y_real[1]}")
            print("--------------------------------------------------------------------------------------------------------")

    # Calculate average error over all samples
    total_error /= len(X)
    if output:
        print(f"Error: {total_error} | Metric: {metrics}")

# Plot the error over training epochs
def plot_error(errors):   
    fig, ax = plt.subplots()
    
    ax.plot(errors, 'blue')
    ax.set_title('Training Error Over Iterations') 
    ax.set_ylabel('Error') 
    ax.set_xlabel('Iteration')
    
    plt.show()

# Number of samples
size = 100
# Number of training samples
train_set_size = 80
# Number of neurons in the hidden layer
hidden_neurons = 30
# Learning rate
learning_rate = 0.01 

# Initialize the input values for each sample
X = np.zeros((size, 3))
X[0] = [8, 5, 3]
step = 0.5
for i in range(1, size):
    if i % 7 == 1:
        X[i] = X[0] + np.array([step, 0, 0])
    elif i % 7 == 2:
        X[i] = X[0] + np.array([0, step, 0])
    elif i % 7 == 3:
        X[i] = X[0] + np.array([0, 0, step])
    elif i % 7 == 4:
        X[i] = X[0] + np.array([-step, 0, 0])
    elif i % 7 == 5:
        X[i] = X[0] + np.array([0, -step, 0])
    elif i % 7 == 6:
        X[i] = X[0] + np.array([0, 0, -step])
    else:
        X[i] = X[0] + np.array([-step, 0, -step])
        step += 1
        i -= 1

# Split the dataset into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_validation_split(X_values=X, train_size=train_set_size, to_normalize=True)

# Train the model using backpropagation
weights_l1, weights_l2, errors_array = backpropagation(X=X_train, Y=Y_train, hidden_neurons=hidden_neurons, lr_rate=learning_rate, threshold=0.01, total_epochs=5000, metric="MSE") 

# Test the model on the testing data 
test_model(X=X_valid, Y=Y_valid, weights_l1=weights_l1, weights_l2=weights_l2, output=True, metrics="MAPE") 

# Plot the training errors on each iteration
plot_error(errors_array)
