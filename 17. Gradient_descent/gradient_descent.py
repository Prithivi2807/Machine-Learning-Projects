import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 100
    n = len(x)
    learning_rate = 0.08     # paratemeter start with some value trial and error based on algorithm parameters

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr #y = mx + b
        # calculate m and b derivative
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])  # mse(cost function) = 1/n n(sum)i=1 (yi-ypredicted)^2
        md = -(2/n)*sum(x*(y-y_predicted)) # 2/n n(sum)i=1 -x(y-(mxi+b))
        bd = -(2/n)*sum(y-y_predicted)     # 2/n n(sum)i=1 -(y-(mxi+b))
        m_curr = m_curr - learning_rate * md    # m = m - learning rate * d/dm
        b_curr = b_curr - learning_rate * bd    # b = b - learning rate * d/db
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)