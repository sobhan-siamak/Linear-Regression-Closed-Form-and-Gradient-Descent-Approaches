


# @copy by sobhan siamak

from numpy import *
import matplotlib.pyplot as plt

# y = mx + b  or  y = theta1*x + theta0
# m or theta1 is slope, b or theta0 is y-intercept

#Calculate Error
def MSE(theta0, theta1, Dataset):
    tError = 0
    for i in range(0, len(Dataset)):
        x = Dataset[i, 0]
        y = Dataset[i, 1]
        tError += (y - (theta1 * x + theta0)) ** 2
    return tError /(float(len(Dataset)))

def step_gradient(b_current, m_current, Dataset, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(Dataset))
    for i in range(0, len(Dataset)):
        x = Dataset[i, 0]
        y = Dataset[i, 1]
        b_gradient = b_gradient - (1/N) * (y - ((m_current * x) + b_current))
        m_gradient = m_gradient - (1/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(Dataset, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    costf = []
    for i in range(num_iterations):
        costf.append(MSE(b, m, Dataset))
        b, m = step_gradient(b, m, array(Dataset), learning_rate)
    return [b, m, costf]

def execute():
    train = genfromtxt("train.csv", delimiter=",", skip_header=True)
    test = genfromtxt("test.csv", delimiter=",", skip_header=True)
    learning_rate = 0.0005
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    # print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, train)))
    # print("Running...")
    [b, m, costf] = gradient_descent_runner(train, initial_b, initial_m, learning_rate, num_iterations)
    theta0 = b
    theta1 = m
    print("After {0} iterations theta0 = {1}, theta1 = {2}, TrainError = {3}, TestError = {4}".format(num_iterations, theta0, theta1, MSE(theta0, theta1, train), MSE(theta0, theta1, test)))


    train2 = genfromtxt('train.csv', delimiter=',', skip_header=True)
    x = array(train2[:, 0])
    x1 = x.reshape(len(x), 1)
    # x2 = np.insert(x1, 0, 1, axis=1)  # add one columns of 1 as bias
    y = array(train2[:, 1])

    plt.scatter(x, y, label='Dataset')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gradient Descent Regression Line:')
    # plt.show()

    yhat = x.dot(m)+b
    plt.plot(x, yhat,label='Regression-Line',color='r')
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(costf)
    plt.title('Cost per iteration')
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost")
    plt.show()
    # # plt.xlabel("No. of iterations")
    # # plt.ylabel('Cost')
    # # plt.title('Cost per iteration')
    # plt.show()




if __name__ == '__main__':
    execute()