import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


n_samples = 100
n_features = 50
n_classes = 2
fea = np.random.rand(n_samples, (n_features,255)) 
print("shape", fea.shape)
labels = np.random.randint(0, n_classes, size=n_samples) 


import numpy as np
import matplotlib.pyplot as plt

def draw_pictures(fea, tsne, labels):

    tsne_results = tsne.fit_transform(fea)

    print("--")
    cmap = plt.get_cmap('tab10')
    

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    max_label = unique_labels.max() 
    
    for label in unique_labels:
  
        indices = labels == label
  
        color = cmap(label / max_label)
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                    s=10, label=f'Label {label}', color=color, alpha=0.6)

    plt.title('t-SNE Visualization with Different Colors for Each Label')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend() 
    plt.savefig("tsne_visualization.png")
    plt.show()

tsne = TSNE(n_components=2, random_state=42)

draw_pictures(fea, tsne, labels)
