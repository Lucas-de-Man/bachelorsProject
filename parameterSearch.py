from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt
from main import Model

with open('music/music.npy', 'rb') as f:
    piano = np.load(f)
    violin = np.load(f)

def black_box_function(x):
    lr = np.power(10, x)
    model = Model(128, lr=lr)
    length = 4096 + model.windowsize
    for _ in range(300):
        p = np.random.randint(0, len(piano) - length)
        v = np.random.randint(0, len(violin) - length)
        losses = model.train(p, v, length)
    score = (sum(losses[0][-400:]) + sum(losses[1][-400:])) / 200
    return score

def nextChoice(model, range, t):
    y_pred, y_std = gp_model.predict(x_range.reshape(-1, 1), return_std=True)
    return np.argmin(y_pred - y_std * np.log(t + 1))

# range of x values
x_range = np.linspace(-5, -1, 1000)

# output for each x value
#black_box_output = black_box_function(x_range)

# random x values for sampling
sample_x = np.random.choice(x_range, size=1)
sample_y = np.array([black_box_function(sample_x)])


kernel = RBF(length_scale=0.05)
gp_model = GaussianProcessRegressor(kernel=kernel)

for t in range(20):
    print(sample_x[-1])
    gp_model.fit(sample_x.reshape(-1, 1), sample_y)

    sample_x = np.append(sample_x, x_range[nextChoice(gp_model, x_range, t)])
    sample_y = np.append(sample_y, black_box_function(sample_x[-1]))

print(sample_x[-1])
gp_model.fit(sample_x.reshape(-1, 1), sample_y)

# Generate predictions using the Gaussian process model
y_pred, y_std = gp_model.predict(x_range.reshape(-1, 1), return_std=True)

# Plot
plt.figure(figsize=(10, 6))
#plt.plot(x_range, black_box_function(x_range), label='Black Box Function')
plt.scatter(sample_x, sample_y, color='red', label='Samples')
plt.plot(x_range, y_pred, color='blue', label='Gaussian Process')
plt.fill_between(x_range, y_pred - 2*y_std, y_pred + 2*y_std, color='blue', alpha=0.2)
plt.xlabel('x')
plt.ylabel('Black Box Output')
plt.title('Black Box Function with Gaussian Process Surrogate Model')
plt.legend()
plt.show()