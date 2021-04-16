### ML - DeepNetwork

This program implements a **_deep neural network_** with one hidden layer that can train on non-linearly separable data.

#### How to start:
1. Initialize a DeepNetwork instance

    - ```deep_network_ = DeepNetwork(size=100)```  
    For randomly generating data set of size 100. 
    
    Data is normally distributed with parameters _N(0, 0.4)_ in the [-1,1] x [-1,1] square.
    There are two subclasses, one - within the 0.5 radius circle, other - outside. 
   
    - ```deep_network = DeepNetwork(filepath='data/data.txt')```  
    Will read data from the data.txt file
    
    Line format for the file is following:  
    `0.700,0.882,0`  
    `0.700 - x-coordinate, 0.882 - y-coordinate, 0 or 1 - class`
    
2. Train the deep network
    - ```deep_network.train()```  
    Splits data in 80/20 for train/test respectively.  
    Trains the network on the submitted data and verifies on test set.  
    After the training has finished, the plots will be generated automatically, including loss function graph, 
    train/test sets and decision boundary.
    
3. Predict for your own data
    - ```deep_network.predict_from_file('data/predict.txt')```  
    Will read data from the _data/predict.txt_ file and predict.  
    The output is written to console.  
    
4. Investigate the reports
    - Concise reports are generated under the `/reports` directory upon each training.
    
    
    
