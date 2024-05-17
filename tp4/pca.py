import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'tp4/europe.csv'
data = pd.read_csv(file_path)

# Drop the 'Country' column for PCA and standardize the data
data_features = data.drop(columns=['Country'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_features)

# Perform PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# Add PCA components to the dataframe
data['PC1'] = pca_components[:, 0]
data['PC2'] = pca_components[:, 1]

# Boxplot of the variables
plt.figure(figsize=(12, 8))
sns.boxplot(data=pd.DataFrame(scaled_data, columns=data_features.columns))
plt.xticks(rotation=45)
plt.title('Boxplot of Economic, Social, and Geographic Variables')
plt.show()

# Define a custom palette with specified colors and lighter shades
custom_palette = [
    "#FF0000", "#FF6347",  # Red, Light Coral
    "#FF4500", "#FFA07A",  # OrangeRed, Light Salmon
    "#FFA500", "#FFDAB9",  # Orange, PeachPuff
    "#FFFF00", "#FFD700",  # Yellow, Light Goldenrod Yellow (adjusted)
    "#7CFC00", "#ADFF2F",  # LawnGreen, GreenYellow (adjusted)
    "#00FF00", "#7FFF00",  # Lime, Chartreuse
    "#20B2AA", "#7FFFD4",  # LightSeaGreen, Aquamarine
    "#00FFFF", "#AFEEEE",  # Aqua, PaleTurquoise
    "#0000FF", "#87CEEB",  # Blue, SkyBlue (adjusted)
    "#8A2BE2", "#9370DB",  # BlueViolet, MediumPurple (adjusted)
    "#FF00FF", "#DA70D6",  # Magenta, Orchid
    "#FF1493", "#FFB6C1",  # DeepPink, LightPink
    "#8B4513", "#F5DEB3",  # SaddleBrown, Wheat
    "#000000", "#D3D3D3"   # Black, LightGrey
]

# Ensure the palette has enough unique colors
if len(custom_palette) < len(data['Country'].unique()):
    raise ValueError("The number of unique colors in the palette is insufficient")

# Biplot of the first two principal components
plt.figure(figsize=(12, 8))

# Scatter plot of the PCA components with distinct colors and markers
sns.scatterplot(x='PC1', y='PC2', data=data, hue='Country', palette=custom_palette, style='Country', s=150)  # Ajuste del tamaño de las formas

# Adding vectors for each feature
for i, feature in enumerate(data_features.columns):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='red', alpha=0.5)
    plt.text(pca.components_[0, i], pca.components_[1, i], feature, color='black')

plt.title('Biplot of the First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Ajustar los márgenes
plt.show()

# Plot the PC1 data for each country as a bar plot
plt.figure(figsize=(14, 8))
plt.bar(data['Country'], data['PC1'], color='orange')
plt.xticks(rotation=90)
plt.title('PC1 per Country')
plt.xlabel('Country')
plt.ylabel('PC1')
plt.tight_layout()  # Ajustar los márgenes
plt.show()
