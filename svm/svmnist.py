import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.manifold import TSNE
import struct
import numpy as np

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

train_pathi = '/home/ross/Coding/ML_course/mnistdb/trainimages'
train_pathl = '/home/ross/Coding/ML_course/mnistdb/trainlabels'
test_pathi = '/home/ross/Coding/ML_course/mnistdb/testimages'
test_pathl = '/home/ross/Coding/ML_course/mnistdb/testlabels'

n_samples = 1000
train_images = extract_images(train_pathi)
train_labels = extract_labels(train_pathl)[:n_samples]
test_images = extract_images(test_pathi)
test_labels = extract_labels(test_pathl)[:n_samples]
train_flat_img = np.array(flatten_images(train_images)[:n_samples])
test_flat_img = np.array(flatten_images(test_images)[:n_samples])


scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(train_flat_img)
X_test_scaled = scaler.transform(test_flat_img)


##########################
## naive svm only approach
##########################
svc = svm.SVC(kernel='linear', decision_function_shape='ovo')
svc.fit(X_train_scaled, train_labels)
predicted = svc.predict(X_test_scaled)
print(predicted)
print(test_labels)
print([int(abs(predicted[i]-test_labels[i])) for i in range(len(predicted))])




#########################
## tsne then svm approach
#########################
embedder = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)
X_train_embedded = embedder.fit_transform(X_train_scaled)
X_test_embedded = embedder.fit_transform(X_test_scaled)

svc.fit(X_train_embedded, train_labels)
predicted = svc.predict(X_test_embedded)


fig = plt.figure()
subfigs = fig.subfigures(1,3)
axL = subfigs[0].subplots(1,1)
subfigs[0].suptitle('train tsne')
axL.scatter(X_train_embedded[:,0], X_train_embedded[:,1] , c=train_labels)
axM = subfigs[1].subplots(1,1)
subfigs[1].suptitle('test fittted with train')
axM.scatter(X_test_embedded[:,0], X_test_embedded[:,1] , c=predicted)
axR = subfigs[2].subplots(1,1)
subfigs[2].suptitle('test actual labels')
axR.scatter(X_test_embedded[:,0], X_test_embedded[:,1] , c=test_labels)
fig.suptitle('diferent manifolds')
plt.show()







