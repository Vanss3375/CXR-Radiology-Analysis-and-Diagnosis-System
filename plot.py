import pandas as pd
import matplotlib.pyplot as plt
import os

GLOBAL_PATH = r'lokasi folder tempat file CSV disimpan'

# Update the path to the correct location of the CSV file
file_path = os.path.join(GLOBAL_PATH, 'volumes.csv')

# Load the CSV file
data = pd.read_csv(file_path)

# Extract the class name from the 'Image Path'
data['Class'] = data['Image Path'].apply(lambda x: x.split('\\')[0])

# Extract features and labels
left_lobe_features = data[['Left Lung Size', 'Left Lung Color']].values
right_lobe_features = data[['Right Lung Size', 'Right Lung Color']].values
labels = data['Class'].values

# Define colors for the classes
unique_classes = data['Class'].unique()
class_colors = {cls: plt.cm.tab10(i) for i, cls in enumerate(unique_classes)}

# Define function to lighten or darken colors
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

# Assign colors based on classes
colors = [class_colors[label] for label in labels]

# Adjust colors for left and right lobes
left_colors = [adjust_lightness(color, 1.2) for color in colors]
right_colors = [adjust_lightness(color, 0.8) for color in colors]

# Plotting
plt.figure(figsize=(10, 6))
for i in range(len(left_lobe_features)):
    plt.scatter(left_lobe_features[i, 0], left_lobe_features[i, 1], color=left_colors[i], label=labels[i] if i == 0 else "", alpha=0.6, edgecolors='w', s=100, marker='o')
    plt.scatter(right_lobe_features[i, 0], right_lobe_features[i, 1], color=right_colors[i], label=labels[i] if i == 0 else "", alpha=0.6, edgecolors='k', s=100, marker='x')

plt.title('Lung Volume and Color for Different Conditions')
plt.xlabel('Volume')
plt.ylabel('Color')
plt.legend(loc='best')
plt.show()
