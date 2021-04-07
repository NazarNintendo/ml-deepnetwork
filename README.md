### ML - Perceptron

This small program is a basic implementation of a simple **_perceptron_** that can train on linearly-separable data.

#### How to start:
1. Initialize a Perceptron instance

    - ```perceptron = Perceptron(size=100)```  
    For randomly generating data set of size 100. 
    
    Data is uniformly distributed in the unit square and is separated by randomly generated line.
   
    - ```perceptron = Perceptron(filepath='data.txt')```  
    Will read data from the data.txt file
    
    Line format for the file is following:  
    `0.700,0.882,0`  
    `0.700 - x-coordinate, 0.882 - y-coordinate, 0 - class`
    
2. Train the perceptron
    - ```perceptron.train()```  
    Splits data in 80/20 for train/test respectively.  
    Trains the perceptron on the submitted data and verifies on test set.  
    After the perceptron has finished, the plots will be generated automatically.
    
3. Predict for your own data
    - ```perceptron.predict(filepath='predict.txt')```  
    Will read data from the predict.txt file and predict.  
    `filepath` parameter is optional, if not provided - defaults to 'predict.txt'
    The output is written to console.  
    
4. Investigate the reports
    - Concise reports are generated under the `/reports` directory upon each training.
    
    
    
