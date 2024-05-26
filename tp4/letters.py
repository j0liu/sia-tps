import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

letters = {
    'A': np.array([
         1,  1,  1,  1,  1,
         1, -1, -1, -1,  1,
         1,  1,  1,  1,  1,
         1, -1, -1, -1,  1,
         1, -1, -1, -1,  1
    ]),
    'B': np.array([
        1,  1,  1,  1, -1,
        1, -1, -1, -1,  1,
        1,  1,  1,  1, -1,
        1, -1, -1, -1,  1,
        1,  1,  1,  1, -1
    ]),
    'C': np.array([
         1,  1,  1,  1,  1,
         1, -1, -1, -1, -1,
         1, -1, -1, -1, -1,
         1, -1, -1, -1, -1,
         1,  1,  1,  1,  1
    ]),
    'D': np.array([
         1,  1,  1,  1, -1,
         1, -1, -1, -1,  1,
         1, -1, -1, -1,  1,
         1, -1, -1, -1,  1,
         1,  1,  1,  1, -1
    ]),
    'E': np.array([
         1,  1,  1,  1,  1,
         1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,
         1, -1, -1, -1, -1,
         1,  1,  1,  1,  1
    ]),
    'F': np.array([
         1,  1,  1,  1,  1,
         1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,
         1, -1, -1, -1, -1,
         1, -1, -1, -1, -1
    ]),
    'G': np.array([
         1,  1,  1,  1,  1,
         1, -1, -1, -1, -1,
         1, -1,  1,  1,  1,
         1, -1, -1, -1,  1,
         1,  1,  1,  1,  1
    ]),
    'H': np.array([
         1, -1, -1, -1,  1,
         1, -1, -1, -1,  1,
         1,  1,  1,  1,  1,
         1, -1, -1, -1,  1,
         1, -1, -1, -1,  1
    ]),
    'I': np.array([
         1,  1,  1,  1,  1,
        -1, -1,  1, -1, -1,
        -1, -1,  1, -1, -1,
        -1, -1,  1, -1, -1,
         1,  1,  1,  1,  1
    ]),
    'J': np.array([
         1,  1,  1,  1,  1,
        -1, -1, -1,  1, -1,
        -1, -1, -1,  1, -1,
         1, -1, -1,  1, -1,
         1,  1,  1, -1, -1
    ]),
    'K': np.array([
        1, -1, -1,  1,  1,
        1, -1,  1, -1, -1,
        1,  1, -1, -1, -1,
        1, -1,  1, -1, -1,
        1, -1, -1,  1,  1
    ]),
    'L': np.array([
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1,  1,  1,  1,  1
    ]),
    'M': np.array([
        1,  1, -1,  1,  1,
        1, -1,  1, -1,  1,
        1, -1,  1, -1,  1,
        1, -1, -1, -1,  1,
        1, -1, -1, -1,  1
    ]),
    'N': np.array([
        1, -1, -1, -1,  1,
        1,  1, -1, -1,  1,
        1, -1,  1, -1,  1,
        1, -1, -1,  1,  1,
        1, -1, -1, -1,  1
    ]),
    'O': np.array([
        1,  1,  1,  1,  1,
        1, -1, -1, -1,  1,
        1, -1, -1, -1,  1,
        1, -1, -1, -1,  1,
        1,  1,  1,  1,  1
    ]),
    'P': np.array([
        1,  1,  1,  1, -1,
        1, -1, -1, -1,  1,
        1,  1,  1,  1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1
    ]),
    'Q': np.array([
         1,  1,  1,  1, -1,
         1, -1, -1,  1, -1,
         1, -1, -1,  1, -1,
         1,  1,  1,  1, -1,
        -1, -1, -1, -1,  1
    ]),
    'R': np.array([
        1,  1,  1,  1, -1,
        1, -1, -1, -1,  1,
        1,  1,  1,  1, -1,
        1, -1, -1,  1, -1,
        1, -1, -1, -1,  1
    ]),
    'S': np.array([
         1,  1,  1,  1,  1,
         1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,
        -1, -1, -1, -1,  1,
         1,  1,  1,  1,  1
    ]),
    'T': np.array([
         1,  1,  1,  1,  1,
        -1, -1,  1, -1, -1,
        -1, -1,  1, -1, -1,
        -1, -1,  1, -1, -1,
        -1, -1,  1, -1, -1
    ]),
    'U': np.array([
        1, -1, -1, -1,  1,
        1, -1, -1, -1,  1,
         1, -1, -1, -1,  1,
         1, -1, -1, -1,  1,
         1,  1,  1,  1,  1
    ]),
    'V': np.array([
          1, -1, -1, -1,  1,
          1, -1, -1, -1,  1,
          1, -1, -1, -1,  1,
         -1,  1, -1,  1, -1,
         -1, -1,  1, -1, -1
    ]),
    'W': np.array([
        1, -1, -1, -1,  1,
        1, -1, -1, -1,  1,
        1, -1,  1, -1,  1,
        1, -1,  1, -1,  1,
        1,  1, -1,  1,  1
    ]),
    'X': np.array([
          1, -1, -1, -1,  1,
         -1,  1, -1,  1, -1,
         -1, -1,  1, -1, -1,
         -1,  1, -1,  1, -1,
          1, -1, -1, -1,  1
    ]),
    'Y': np.array([
         1, -1, -1, -1,  1,
         1, -1, -1, -1,  1,
         1,  1,  1,  1,  1,
        -1, -1, -1, -1,  1,
         1,  1,  1,  1,  1
    ]),
    'Z': np.array([
         1,  1,  1,  1,  1,
        -1, -1, -1,  1, -1,
        -1, -1,  1, -1, -1,
        -1,  1, -1, -1, -1,
         1,  1,  1,  1,  1
    ])
}


def get_letter(letter):
    return letters[letter]

def add_noise(letter, noise_level):
    noisy_letter = get_letter(letter).copy()
    for i in range(len(noisy_letter)):
        if np.random.rand() < noise_level:
            noisy_letter[i] *= -1
    return noisy_letter

def plot_single_pattern(pattern, label):
    plt.figure()
    plt.imshow(pattern.reshape(5, 5), cmap='binary')
    plt.title(f'Letter {label}')
    plt.axis('off')
    plt.show()

def plot_all_patterns_together(patterns, labels):
    plt.figure(figsize=(15, 10))
    grid_size = int(np.ceil(np.sqrt(len(patterns))))

    for idx, (letter, pattern) in enumerate(zip(labels,patterns), 1):
        plt.subplot(grid_size, grid_size, idx)
        plt.imshow(pattern.reshape(5, 5), cmap='binary')
        plt.title(letter)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def analyze_groups(letters):
    flat_letters = {k: m.flatten() for k, m in letters.items()}
    all_groups = itertools.combinations(flat_letters.keys(), 4)

    avg_dot_product_list = []
    max_dot_product_list = []

    for g in all_groups:
        group = np.array([v for k, v in flat_letters.items() if k in g])
        orto_matrix = group.dot(group.T)
        np.fill_diagonal(orto_matrix, 0)
        row, _ = orto_matrix.shape
        avg_dot_product = np.abs(orto_matrix).sum()/(orto_matrix.size-row)
        avg_dot_product_list.append((avg_dot_product, g))
        max_v = np.abs(orto_matrix).max()

        max_poses = np.argwhere(np.abs(orto_matrix) == max_v)
        max_poses = list(set([tuple(sorted(p)) for p in max_poses]))

        # max_pos = np.unravel_index(np.argmax(np.abs(orto_matrix), axis=None), orto_matrix.shape)
        max_dot_product_count = np.count_nonzero(np.abs(orto_matrix) == max_v) / 2
        max_dot_product_list.append((max_v, max_dot_product_count, g, max_poses))

    avg_dot_product_sorted = sorted(avg_dot_product_list, key=lambda x: x[0])
    max_dot_product_sorted = sorted(max_dot_product_list, key=lambda x: (x[0], x[1]))

    df_avg = pd.DataFrame(avg_dot_product_sorted, columns=["Average", "Group"])
    df_max = pd.DataFrame(max_dot_product_sorted, columns=["Max", "Count", "Group", "Pos Max"])


    # Merge the two dataframes to get a combined view
    df_combined = pd.merge(df_avg, df_max, on="Group")
    with open('tp4/plots/hopfield/archive/groups.txt', 'w') as f:
        for g, avg_dot_product, max_v, max_dot_product_count in zip(df_combined['Group'], df_combined['Average'], df_combined['Max'], df_combined['Count']):
            f.write(f'{g} {avg_dot_product:.1f} {max_v:.1f} {max_dot_product_count}\n')

    # Display the best and worst 4 groups
    print("Best 4 Groups by Average Dot Product:")
    print(df_combined.head(10))

    print(df_combined.iloc[len(df_avg) // 2])

    print("\nWorst 4 Groups by Average Dot Product:")
    print(df_combined.tail(10))

def export_pattern(pattern):
    with open('tp4/plots/hopfield/archive/pattern.txt', 'a') as f:
        f.write(' '.join([str(int(i)) for i in pattern]))
        f.write('\n')

if __name__ == '__main__':
    plot_all_patterns_together(letters.values(), letters.keys())
    analyze_groups(letters)
