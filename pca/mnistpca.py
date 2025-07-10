import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD



#importing the data
import struct

def extract_images(file_path):
    with open(file_path, 'rb') as file:
        magic_number = struct.unpack('>I', file.read(4))[0]
        num_images = struct.unpack('>I', file.read(4))[0]
        num_rows = struct.unpack('>I', file.read(4))[0]
        num_cols = struct.unpack('>I', file.read(4))[0]
        
        images = []
        for _ in range(num_images):
            image = []
            for _ in range(num_rows):
                row = []
                for _ in range(num_cols):
                    pixel = struct.unpack('>B', file.read(1))[0]
                    row.append(pixel)
                image.append(row)
            images.append(image)
        
        return images


def flatten_images(list_img):
    flattened_list = []
    for img in list_img:
        tmp=[]
        for line in img:
            tmp+=line
        flattened_list.append(tmp)
    return flattened_list

def extract_labels(file_path):
    with open(file_path, 'rb') as file:
        magic_number = struct.unpack('>I', file.read(4))[0]  # Lire le magic number
        num_labels = struct.unpack('>I', file.read(4))[0]    # Lire le nombre de labels
        
        labels = []
        for _ in range(num_labels):
            label = struct.unpack('>B', file.read(1))[0]      # Lire un label (1 octet)
            labels.append(label)
        
        return labels

pathi = './trainimages'
pathl = './trainlabels'
images = extract_images(pathi)
flat_img = flatten_images(images)
labels = extract_labels(pathl)

#tests unitaires
print(images[0],images[1])
print(labels[0],labels[1])
print(flat_img[0], flat_img[1])
plt.imshow(images[0], cmap='gray')
plt.title(f'Label: {labels[0]}')
plt.savefig('image_mnist.png')  # Sauvegarde l'image dans un fichier
plt.close()


#testing centering and SVD Calculations
x = flat_img #list of vectors in Rd

X = np.transpose(np.array(x))
moy = np.array([np.sum(X,axis=1)/len(x)])
X_moy = np.matmul(moy.T, np.array([[1]*len(x)]))
X_centered = X - X_moy

print(moy, np.shape(moy))
print(X_centered[:,1])

svd = TruncatedSVD(n_components=2, n_iter=10, random_state=5)
U = svd.fit_transform(np.transpose(X_centered))
Sigma = np.diag(svd.singular_values_)
V = svd.components_


print(U)
plt.scatter(U[:,0],U[:,1], c=labels)
plt.savefig('pca_result.png')
plt.close()
