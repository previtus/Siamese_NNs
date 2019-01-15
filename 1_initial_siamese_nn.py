# Initial tests with Siamese CNNs

EPOCHS = 30
SUBSET = 20000
SIZE = [100,100]

# using code from:
# - https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
# - https://sorenbouma.github.io/blog/oneshot/


# We have two datasets, both containing images of the same area in years 2010 and 2013 (old and new)
# File of the same number <n> are the same:
# - 2010_300x300_tiles.tif.<0>.tif
# - tiles_2013.tif.<0>.tif

# 326 files are excluded (they are on the corner of the original image)

from os import listdir
from os.path import isfile, join
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D, Activation, Dropout, ZeroPadding2D
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

path_old = "/scratch/ruzicka/dataset_initial_view/01_10_twoTimes-OrthoAndVector/2010_ortho_2013_vector/300x300_tiles_2010/"
path_new = "/scratch/ruzicka/dataset_initial_view/01_10_twoTimes-OrthoAndVector/2013_ortho_2014_vector/300x300_tiles_2013/"

tile_paths_old = [f for f in listdir(path_old) if isfile(join(path_old, f))]
tile_paths_new = [f for f in listdir(path_new) if isfile(join(path_new, f))]

print("tile_paths_old", len(tile_paths_old), tile_paths_old[0])
print("tile_paths_new", len(tile_paths_new), tile_paths_new[0])

# infra channel is useless!
"""
overall_max = -9999
overall_min = 9999

for i in range(0, len(tile_paths_new)):
    img_old = mpimg.imread(path_new + tile_paths_new[i])
    infra = img_old[:,:,3]

    if overall_max < infra.max():
        overall_max = infra.max()
    if overall_min > infra.min():
        overall_min = infra.min()
    #print(infra.shape, infra.ndim, infra.min(), infra.max())

print("Min and max in the near infra channel:", overall_min, overall_max)
# OLD images: Min and max in the near infra channel: 255 255
# NEW images: Min and max in the near infra channel: 255 255
"""

# show 4 channels:
"""
fig=plt.figure(figsize=(8, 8))
columns = 4 # needs to be multiple of 4
rows = 4
for i in range(1, columns*rows +1, 4):
    idx = i * 20 + random.randint(1,1001)

    img_old = mpimg.imread(path_old + tile_paths_old[idx])
    fig.add_subplot(rows, columns, i)
    plt.imshow(img_old[:,:,0])
    fig.gca().set_axis_off()

    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img_old[:,:,1])
    fig.gca().set_axis_off()

    fig.add_subplot(rows, columns, i+2)
    plt.imshow(img_old[:,:,2])
    fig.gca().set_axis_off()

    fig.add_subplot(rows, columns, i+3)
    plt.imshow(img_old[:,:,3])
    fig.gca().set_axis_off()

plt.subplots_adjust(wspace=0.1, hspace=0.1)
fig.suptitle("Image channels R,G,B, near infra")
plt.show()
"""


# Show pairs
"""
fig=plt.figure(figsize=(8, 8))
columns = 10 # needs to be multiple of 10
rows = 10
#off = random.randint(1,1001)
for i in range(1, columns*rows +1, 2):
    #idx = i * 20 + off
    idx = i * 20 + random.randint(1,1001)

    img_old = mpimg.imread(path_old + tile_paths_old[idx])
    fig.add_subplot(rows, columns, i)
    plt.imshow(img_old)
    fig.gca().set_axis_off()

    img_new = mpimg.imread(path_new + tile_paths_new[idx])
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img_new)
    fig.gca().set_axis_off()

plt.subplots_adjust(wspace=0.1, hspace=0.1)
fig.suptitle("Showing the same space in two points in time")

fig.text(0.5, 0.04, 'Showing pairs: old image (2010), new image (2013)', ha='center')

plt.show()
"""

# Create a dataset of pairs - with labels of same/different
# note that we have very sparce data - we have only one positive per each pair (and posibly all other are negatives)
def create_pairs(listA, listB):
    """
    Create pairs from two lists of image paths. In the same indice the images are the same.
    We alternate between positive and negative examples.

    :param listA: list of images (for example the old ones)
    :param listB: list of images (for example the new ones)
    :return: list of pairs
    """

    pairs = []
    labels = []
    for i in range(0,len(listA)):
        pairs += [[listA[i],listB[i]]] # same

        compare_to = i
        while compare_to == i: #not comparing to itself
            compare_to = random.randint(0,len(listA)-1)

        pairs += [[listA[i], listB[compare_to]]]  # different

        labels += [1, 0]
    return np.array(pairs), np.array(labels)

def load_images_with_keras(img_paths, target_size=None):
    imgs_arr = [img_to_array(load_img(path, target_size=target_size)) for path in img_paths]
    imgs_arr = np.array(imgs_arr)
    return imgs_arr

def load_images_with_matlibplot(img_paths):
    imgs_arr = [mpimg.imread(path) for path in img_paths]
    imgs_arr = np.array(imgs_arr)
    return imgs_arr


def shuffle_two_lists_together(a,b, SEED=None):
    if SEED is not None:
        random.seed(SEED)

    sort_order = random.sample(range(len(a)), len(a))
    a_new = [a[i] for i in sort_order]
    b_new = [b[i] for i in sort_order]
    a_new = np.asarray(a_new)
    b_new = np.asarray(b_new)
    return a_new, b_new

def split_3arr_data(x,y,z,validation_split=0.3, v=True):
    split_at = int(len(x) * (1 - validation_split))
    x_train = x[0:split_at]
    y_train = y[0:split_at]
    z_train = z[0:split_at]
    x_test = x[split_at:]
    y_test = y[split_at:]
    z_test = z[split_at:]

    if v:
        print("Split", len(x), "images into", len(x_train), "train and", len(x_test), "test sets.")
    return x_train,y_train,z_train,x_test,y_test,z_test


pairs, labels = create_pairs(tile_paths_old, tile_paths_new)
# now we can shuffle the two lists, keeping the order
pairs, labels = shuffle_two_lists_together(pairs, labels)

left_input = pairs[:,0]
right_input = pairs[:,1]

# add the full path
left_input = np.array([path_old + i for i in left_input])
right_input = np.array([path_new + i for i in right_input])

print(left_input.shape)
print(right_input.shape)

# -----------------------------------------------------------
# SLOW LOADING OF IMAGES
left_input = left_input[0:SUBSET]
right_input = right_input[0:SUBSET]
labels = labels[0:SUBSET]
# -----------------------------------------------------------

print("left_input", len(left_input), left_input[0])
print("right_input", len(right_input), right_input[0])
print("same 1 / different 0:", labels[0])

# version A
#left_input = load_images_with_matlibplot(left_input)
#right_input = load_images_with_matlibplot(right_input)

# version B
left_input = load_images_with_keras(left_input,target_size=SIZE)
right_input = load_images_with_keras(right_input,target_size=SIZE)
left_input = left_input.astype('float32')
right_input = right_input.astype('float32')
left_input /= 255
right_input /= 255

input_shape = left_input[0].shape

print("left_input", len(left_input), left_input[0].shape)
print("right_input", len(right_input), right_input[0].shape)
print("same 1 / different 0:", labels[0])

# show same/different pairs
"""
fig=plt.figure(figsize=(8, 8))
columns = 2 # should be 2
rows = 4
#off = random.randint(1,1001)
for i in range(1, columns*rows +1, 2):
    #idx = i * 20 + off
    idx = random.randint(0,len(left_input)-1)

    img_old = left_input[idx]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img_old)

    txt_labels=["different", "same"]
    fig.gca().set(ylabel=txt_labels[labels[idx]])

    img_new = right_input[idx]
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img_new)

plt.show()
"""

# Split the data into train - test sets
#x_train,y_train,z_train,x_test,y_test,z_test = split_3arr_data(x,y,z)
left_input_train,right_input_train,labels_train,\
left_input_test, right_input_test, labels_test = split_3arr_data(left_input,right_input,labels)


print("left_input_train",left_input_train.shape)
print("right_input_train",right_input_train.shape)
print("labels_train",labels_train.shape)
print("left_input_test",left_input_test.shape)
print("right_input_test",right_input_test.shape)
print("labels_test",labels_test.shape)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    # Bigger model from an example
    # http://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    # https://sorenbouma.github.io/blog/oneshot/
    #Train on 14000 samples, validate on 6000 samples
    #14000/14000 [==============================] - 901s 64ms/step - loss: 0.0733 - accuracy: 0.9395 - val_loss: 0.0805 - val_accuracy: 0.9257
    #* Accuracy on training set: 93.08%
    #* Accuracy on test set: 92.57%

    def W_init(shape, name=None):
        """Initialize weights as in paper"""
        values = np.random.normal(loc=0, scale=1e-2, size=shape)
        return K.variable(values, name=name)

    # //TODO: figure out how to initialize layer biases in keras.
    def b_init(shape, name=None):
        """Initialize bias as in paper"""
        values = np.random.normal(loc=0.5, scale=1e-2, size=shape)
        return K.variable(values, name=name)

    convnet = Sequential()
    convnet.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                       kernel_initializer=W_init, kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (7, 7), activation='relu',
                       kernel_regularizer=l2(2e-4), kernel_initializer=W_init, bias_initializer=b_init))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=W_init, kernel_regularizer=l2(2e-4),
                       bias_initializer=b_init))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=W_init, kernel_regularizer=l2(2e-4),
                       bias_initializer=b_init))
    convnet.add(Flatten())
    convnet.add(Dense(4096, activation="sigmoid", kernel_regularizer=l2(1e-3), kernel_initializer=W_init,
                      bias_initializer=b_init))
    return convnet

    # alternatives:
    """
    # With some CONV
    convnet = Sequential([
        Conv2D(kernel_size = (5,5), filters = 20, input_shape=input_shape),
        Activation('relu'),
        MaxPooling2D(),
        #Conv2D(kernel_size = (5,5), filters = 25),
        #Activation('relu'),
        #MaxPooling2D(),
        Flatten(),
        Dense(128),
        Dropout(0.1),
        Dense(128),
        Dropout(0.1),
        Dense(128),
    ])
    return convnet

    # Tiny one, made sense for MNIST, also Dense
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)
    """

# Methods from the Keras sample:
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))



# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)


# train
rms = RMSprop(lr=0.0001)
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
history = model.fit([left_input_train, right_input_train], labels_train,
          batch_size=128,
          epochs=EPOCHS,
          validation_data=([left_input_test, right_input_test], labels_test))

# compute final accuracy on training and test sets
y_pred = model.predict([left_input_train, right_input_train])
tr_acc = compute_accuracy(labels_train, y_pred)
y_pred = model.predict([left_input_test, right_input_test])
te_acc = compute_accuracy(labels_test, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

from visualize_history_pretty import nice_plot_history
nice_plot_history(history)