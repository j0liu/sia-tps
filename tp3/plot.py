import matplotlib.pyplot as plt
import numpy as np
import csv 

with open("tp3/data.csv") as f:
    data = list(csv.reader(f)) 
    data = np.array(data[1:], dtype=float)

def plot(data):
    ax = plt.figure().add_subplot(projection='3d')

    # Plot a sin curve using the x and y axes.
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    result = data[:, 3]

    ax.scatter(x, y, zs=0, zdir='z', label='curve in (x, y)')

    # Plot scatterplot data (20 2D points per colour) on the x and z axes.
    colors = ('r', 'g', 'b', 'k')

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    c_list = colors * (len(x) // len(colors)) + colors[:len(x) % len(colors)]
    # By using zdir='y', the y value of these points is fixed to the zs value 0
    # and the (x, y) points are plotted on the x and z axes.
    ax.scatter(x, y, zs=result, zdir='z', c=c_list)

    # # Make legend, set axes limits and labels
    ax.legend()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=20., azim=-35)
    plt.show()

def plotxy(inputs, expected, w, delta_w):
    ax = plt.figure().add_subplot()

    # Plot a sin curve using the x and y axes.
    ax.scatter(inputs, expected, label='expected')
    # for l in outputs:
        #random color for the line
    colors = np.random.rand(1)
    x = np.linspace(-10, 20, 100)
    prev = w - delta_w
    ax.plot(x, w[1]*x + w[0], label='w')
    ax.plot(x, delta_w[1]*x + delta_w[0], label='delta_w')
    ax.plot(x, prev[1]*x + prev[0], label='prev')
    # # Make legend, set axes limits and labels
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    plt.show()