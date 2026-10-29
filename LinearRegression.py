import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
y = np.array([4.1, 7.2, 9.8, 13.1, 15.9, 19.2, 22.0, 25.1, 28.0, 31.2])

def NNloss(W,b,x,y):
    y_pred = W*x + b
    return ((y_pred - y)**2).sum()

def grad(W,b,x,y):
    y_pred = W*x + b
    
    Grad_W = (2*(y_pred - y)*x).sum()
    Grad_b = 2*(y_pred - y).sum()
    return Grad_W, Grad_b

def LinearRegression(x,y, epochs = 50000):
    W = 0.0
    b = 0.0
    lr = 0.0001
    
    for i in range(epochs):
        Grad_W, Grad_B = grad(W,b,x,y)
        W -= Grad_W*lr
        b -= Grad_B*lr
    
    return {
        'W':W,
        'b':b,
    }
    
Dict = LinearRegression(x,y)

print(Dict['W'], Dict['b'], sep = "\n")

 
     
    
        
    
    
    