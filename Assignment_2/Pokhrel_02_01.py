# Pokhrel, Apar
# 1001_646_558
# 2024_10_06
# Assignment_02_01


import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

'''
Create a MultiNN class

'''

class MultiNN(nn.Module):
    def __init__(self,  layers, activations):
        super(MultiNN, self).__init__()
        
        self.layers = layers
        self.activations = activations
        self.weights = nn.ParameterList() #https://pytorch.org/docs/stable/generated/torch.nn.ParameterList.html
        
    def apply_activation_function(self, X, activation_function):
        activation_function = activation_function.lower()
        
        match activation_function:
            case 'linear':
                return X
            
            case 'sigmoid':
                return 1 / (1 + torch.exp(-X))
            
            case 'relu':
               return torch.max(torch.tensor(0.0), X)
            
        
      
    '''
        Referenced from Assignment 1.
        
        initializes weights given a # of nodes in a layer or the numpy weight matrix
    '''
    def initialize_weights(self, X, seed):
        weights = []
        
        if all(isinstance(layer, int) for layer in self.layers):
            input_dims = X.shape[1]
            for i in range(len(self.layers)):
                input_size = self.layers[i-1] if i > 0 else input_dims
                output_size = self.layers[i]
                np.random.seed(seed)
                W = np.random.randn(input_size + 1, output_size).astype(np.float32)  # adding +1 for the bias
                
                # convert into tensort and add coomputational graph 
                weights.append(torch.tensor(W, dtype=torch.float32, requires_grad=True))
                
        elif all(isinstance(layer, np.ndarray) for layer in self.layers):
            # initialize the weights as it it and convert to tensor, and add computational graph
            weights = [torch.tensor(w, dtype=torch.float32, requires_grad=True) for w in self.layers]
        
        self.weights.extend(nn.Parameter(w) for w in weights)
        return self.weights
    
    
    '''
        Referenced from Assignment 1 
        
        Calculates the forward pass of the network with it's activation function
    
    '''
    def calculate_output(self, X):
        
        # add ones
        X = torch.cat((torch.ones((X.shape[0], 1), dtype=torch.float32), X), dim=1)
    

        for i in range(len(self.weights)):
            result = torch.mm(X, self.weights[i])  # dot product with weight matrix
            X = self.apply_activation_function(result, self.activations[i])

            if i < len(self.weights) - 1:  # keep stacking ones unitl the last layer
                X = torch.cat((torch.ones((X.shape[0], 1), dtype=torch.float32), X), dim=1)
        return X

'''
    Calculates mean absolute error btw target and predicted output
'''
def calculate_mae(target, predicted):
    return torch.mean(torch.abs(target - predicted))


'''
 Applies loss function for the network given the loss function

'''
def apply_loss_function(loss_function, target, output):
    loss_function = loss_function.lower()
    match loss_function:
        case 'mse':
            return torch.mean((target - output) ** 2)
        case 'ce':
            output = torch.softmax(output,dim=1)
            return  -torch.mean(torch.sum(target * torch.log(output), dim=1))
        case 'svm':
            #not working? 
            loss = torch.max(torch.zeros_like(output), output - target + 1)
            return torch.mean(loss)


'''
    Creates validation sets for x_train and y_train. Re-computes the x_train, and y_train by stripping away the validation set
'''
def prepare_data(X, Y, val_split):
    num_samples = X.shape[0]
    
    validation_start_idx = int(val_split[0] * num_samples)  #start_index
    validation_end_idx = int(val_split[1] * num_samples)   #end_index
    
    
    x_val = X[validation_start_idx:validation_end_idx]
    y_val = Y[validation_start_idx:validation_end_idx]
    
    x_new = np.concatenate((X[:validation_start_idx], X[validation_end_idx:]),axis=0)
    
    y_new = np.concatenate((Y[:validation_start_idx], Y[validation_end_idx:]),axis=0)

    return x_new, y_new, x_val, y_val



def multi_layer_nn_torch(
    x_train,
    y_train,
    layers,
    activations,
    alpha=0.01,
    batch_size=32,
    epochs=0,
    loss_func='mse',
    val_split=(0.8, 1.0),
    seed=7321,
):

    model = MultiNN(layers, activations)
    
    #print(model)
    
    model.initialize_weights(torch.tensor(x_train, dtype=torch.float32),seed)
    
    #print('my weights are', model.weights)
    
    x_new, y_new, x_val, y_val = prepare_data(x_train, y_train, val_split)

    #print('type is:' , type(model.weights))
    
    # when no training happens
    if epochs == 0:
        with torch.no_grad():
            x_val_output = model.calculate_output(torch.tensor(x_val, dtype=torch.float32))
            
            # removes the computation graph.
            np_weights = [weight.detach().numpy() for weight in model.weights]
            
            np_x_val_output = x_val_output.detach().numpy()
            return [np_weights, [], np_x_val_output]
    
    # TensorDataset(x_new,y_new)
    train_ds = TensorDataset(torch.tensor(x_new, dtype=torch.float32),torch.tensor(y_new, dtype=torch.float32))
    
        
    optimizer = SGD(model.parameters(), lr=alpha)
    
    # mean abosulte error per epoch
    mae_errors_per_epoch =[]
    
    # for loop based on Google Colab Notebook from class
    for epoch in range(epochs):
        train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False)
        for batch_x,batch_y in train_dl:
            output = model.calculate_output(batch_x)
            
            loss = apply_loss_function(loss_func, batch_y, output)
            #print('my loss is: ',loss)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()

        # https://pytorch.org/docs/stable/generated/torch.no_grad.html
        with torch.no_grad():
            x_val_output = model.calculate_output(torch.tensor(x_val, dtype=torch.float32))
            
            mae_error = calculate_mae(torch.tensor(y_val, dtype=torch.float32),x_val_output)
            mae_errors_per_epoch.append(mae_error)
            
    # again remove computational graph and convert as numpy array
    np_final_weights = [weight.detach().numpy() for weight in model.weights]
 

    return [np_final_weights, mae_errors_per_epoch, x_val_output.detach().numpy()]
    


  