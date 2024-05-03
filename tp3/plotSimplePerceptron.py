import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def plot_decision_boundary(w_list, inputs, expected, title, folder='tp3/plots'):
    # Check if the directory exists, if not, create it
    if not os.path.exists(folder + '/' + title):
        os.makedirs(folder + '/' + title)

    # Clean the folder
    for file in os.listdir(folder + '/' + title):
        os.remove(os.path.join(folder + '/' + title, file))
    
    for index, w in enumerate(w_list):
        # Set up the figure
        fig, ax = plt.subplots()
        
        # Add title
        ax.set_title(title)
        
        #Plot the decision boundary
        x = np.linspace(-2, 2, 100)
        y = -w[1]/w[2] * x - w[0]/w[2]
        ax.plot(x, y, label=f'Decision Boundary {index+1}')
        # Plot input samples (as points)
        for idx, input in enumerate(inputs):
            marker = 'o' if expected[idx] == 1 else 'x'
            ax.scatter(input[1], input[2], marker=marker, c='r' if expected[idx] == 1 else 'b')

        # Additional plot settings
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True)
        ax.set_title('Decision Boundary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Show legend and plot
        ax.legend()

        # Save the figure
        plt.savefig(os.path.join(f'{folder}/{title}/decision_boundary_{index+1:03d}.png'))
        plt.close(fig)  # Close the figure to free up memory
    
    create_gif_from_folder(title, folder, 'decision_boundary_animation.gif', 4)

def create_gif_from_folder(type, folder_path, output_gif_name, duration=500):
    # Ensure the folder exists
    if not os.path.exists(folder_path+'/'+type):
        raise ValueError(f"The specified folder does not exist: {folder_path+'/'+type}")
    
    if not os.path.exists('tp3/plots/gifs/' + type):
        os.makedirs('tp3/plots/gifs/' + type)
    
    # Get all image files in the folder, sorted to maintain order
    files = sorted([os.path.join(folder_path+'/'+type, f) for f in os.listdir(folder_path+'/'+type) if f.endswith('.png')])
    
    # Create an image list
    images = [Image.open(file) for file in files]
    
    # Save the images as a GIF
    images[0].save(f'tp3/plots/gifs/{type}/{output_gif_name}', save_all=True, append_images=images[1:], duration=duration, loop=0)

def plot_errors(errors, type):

    if not os.path.exists('tp3/plots/errors'):
        os.makedirs('tp3/plots/errors')

    fig, ax = plt.subplots()
    ax.plot(errors)
    ax.grid(True)
    ax.set_yscale('linear')
    ax.set_title(f'{type} Errors')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    plt.savefig(f'tp3/plots/errors/{type}.png')
    plt.close(fig)