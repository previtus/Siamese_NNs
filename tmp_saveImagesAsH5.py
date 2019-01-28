SUBSET = 53000
#SUBSET = 100
SIZE = [100,100]

from os import listdir
from os.path import isfile, join
import random
import numpy as np
import matplotlib.image as mpimg
from keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

path_old = "/scratch/ruzicka/dataset_initial_view/01_10_twoTimes-OrthoAndVector/2010_ortho_2013_vector/300x300_tiles_2010/"
path_new = "/scratch/ruzicka/dataset_initial_view/01_10_twoTimes-OrthoAndVector/2013_ortho_2014_vector/300x300_tiles_2013/"

tile_paths_old = [f for f in listdir(path_old) if isfile(join(path_old, f))]
tile_paths_new = [f for f in listdir(path_new) if isfile(join(path_new, f))]

tile_paths_old = tile_paths_old[0:-1]
tile_paths_new = tile_paths_new[0:-1]

print("tile_paths_old", len(tile_paths_old), tile_paths_old[0])
print("tile_paths_new", len(tile_paths_new), tile_paths_new[0])

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
    imgs_arr = []
    for path in tqdm(img_paths):
        imgs_arr.append( img_to_array(load_img(path, target_size=target_size)) )
    #imgs_arr = [img_to_array(load_img(path, target_size=target_size)) for path in img_paths]
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





# =========================

import h5py

prep = "/scratch/ruzicka/dataset_initial_view/01_10_twoTimes-OrthoAndVector/"
hdf5_path = prep+"preloadedImgs_"+str(SUBSET)+"_"+str(SIZE[0])+"x"+str(SIZE[1])+".h5"


hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("left_input", data=left_input, dtype="float32")
hdf5_file.create_dataset("right_input", data=right_input, dtype="float32")
hdf5_file.create_dataset("labels", data=labels, dtype="float32")
hdf5_file.close()