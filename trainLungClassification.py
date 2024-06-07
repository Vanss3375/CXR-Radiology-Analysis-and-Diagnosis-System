import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import joblib

file_path = 'volumes.csv'
data = pd.read_csv(file_path)
data['Class'] = data['Image Path'].apply(lambda x: x.split('\\')[0])
left_lobe_features = data[['Left Lung Size', 'Left Lung Color']].values
right_lobe_features = data[['Right Lung Size', 'Right Lung Color']].values
labels = data['Class'].values
left_lobe_features[:, 0] = np.log1p(left_lobe_features[:, 0])
right_lobe_features[:, 0] = np.log1p(right_lobe_features[:, 0])
features = np.vstack([left_lobe_features, right_lobe_features])
combined_labels = np.hstack([labels, labels])
class_mapping = {label: idx for idx, label in enumerate(np.unique(combined_labels))}
encoded_labels = np.array([class_mapping[label] for label in combined_labels])
scaler = StandardScaler()
features = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.3, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1, warm_start=True, random_state=1)

model_file_path = 'AIs/Classification NN/weights/mlp_model.joblib'
scaler_file_path = 'AIs/Classification NN/weights/scaler.joblib'

num_epochs = 1000
train_loss = []
train_accuracy = []
train_precision = []
train_recall = []
train_map = []

for epoch in range(num_epochs):
    mlp.fit(X_train, y_train)
    
    y_train_pred = mlp.predict(X_train)
    y_train_pred_proba = mlp.predict_proba(X_train)
    
    loss = mlp.loss_
    acc = accuracy_score(y_train, y_train_pred)
    prec = precision_score(y_train, y_train_pred, average='weighted')
    rec = recall_score(y_train, y_train_pred, average='weighted')
    map_score = np.mean([average_precision_score(y_train == i, y_train_pred_proba[:, i]) for i in range(len(class_mapping))])
    
    train_loss.append(loss)
    train_accuracy.append(acc)
    train_precision.append(prec)
    train_recall.append(rec)
    train_map.append(map_score)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} - Accuracy: {acc:.4f} - Precision: {prec:.4f} - Recall: {rec:.4f} - mAP: {map_score:.4f}")

joblib.dump(mlp, model_file_path)
joblib.dump(scaler, scaler_file_path)

print(f"Model saved to {model_file_path}")
print(f"Scaler saved to {scaler_file_path}")

y_pred = mlp.predict(X_test)
report = classification_report(y_test, y_pred, target_names=class_mapping.keys())
print("Classification Report:\n", report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

plt.figure(figsize=(8, 6))
plt.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('AIs/Classification NN/confusion_matrix.png')
plt.close()

num_classes = len(class_mapping)
num_cols = 2
num_rows = (num_classes + 1) // num_cols

plt.figure(figsize=(14, 10))
for i, cls in enumerate(class_mapping.keys()):
    plt.subplot(num_rows, num_cols, i+1)
    precision, recall, _ = precision_recall_curve(y_test == i, mlp.predict_proba(X_test)[:, i])
    plt.plot(recall, precision, marker='.')
    plt.title(f'Precision-Recall curve for {cls}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
plt.tight_layout()
plt.savefig('AIs/Classification NN/precision_recall_curves.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('AIs/Classification NN/training_loss_curve.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(train_accuracy, label='Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('AIs/Classification NN/accuracy_curve.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(train_precision, label='Precision')
plt.title('Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.savefig('AIs/Classification NN/precision_curve.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(train_recall, label='Recall')
plt.title('Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.savefig('AIs/Classification NN/recall_curve.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(train_map, label='mAP')
plt.title('Mean Average Precision (mAP)')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.legend()
plt.savefig('AIs/Classification NN/mAP_curve.png')
plt.close()

x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(plt.cm.tab10.colors))
unique_classes = data['Class'].unique()
class_colors = {cls: plt.cm.tab10(i) for i, cls in enumerate(unique_classes)}
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


colors = [class_colors[label] for label in labels]
left_colors = [adjust_lightness(color, 1.2) for color in colors]
right_colors = [adjust_lightness(color, 0.8) for color in colors]
for i in range(len(left_lobe_features)):
    plt.scatter(features[i, 0], features[i, 1], color=left_colors[i], alpha=0.6, edgecolors='w', s=100, marker='o')
    plt.scatter(features[i + len(left_lobe_features), 0], features[i + len(left_lobe_features), 1], color=right_colors[i], alpha=0.6, edgecolors='k', s=100, marker='x')
patches = [plt.Line2D([0], [0], marker='o', color='w', label=cls, markerfacecolor=class_colors[cls], markersize=10) for cls in unique_classes]
plt.legend(handles=patches, loc='best')
plt.title('Lung Volume and Color for Different Conditions with Decision Boundaries')
plt.xlabel('Lobe Size')
plt.ylabel('Lobe Color')
plt.savefig('AIs/Classification NN/classification.png')
plt.show()
print("All graphs have been saved to the current directory.")