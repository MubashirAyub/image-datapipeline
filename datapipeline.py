import tensorflow as tf
import numpy as np
import time
import os
import glob
import random

from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

file_pattern = r"..\lowsize_8gb_test_db\*\*\file.npy"  # 8 gb database (80mb x 100files) 10485760 shape
# file_pattern = r"..\randomSize_db\*\*\file.npy" # 24 gb database (240mb x 100files) 31457280 shape



#training_files = glob.glob(os.path.join(file_pattern))
def collecting_data(data_files_path, label_location_in_path):
    data_files = glob.glob(os.path.join(data_files_path))
    random.shuffle(data_files)
    return data_files, label_location_in_path


files, labels = collecting_data(file_pattern, -3)

def npy_header_offset(npy_path):
    """Gives the no of header bytes inside a numpy file."""
    with open(str(npy_path), 'rb') as f:
        if f.read(6) != b'\x93NUMPY':
            raise ValueError('Invalid NPY file.')
        version_major, version_minor = f.read(2)
        if version_major == 1:
            header_len_size = 2
        elif version_major == 2:
            header_len_size = 4
        else:
            raise ValueError('Unknown NPY file version {}.{}'.
                             format(version_major, version_minor))
        header_len = sum(b << (8 * i) for i, b in enumerate(
            f.read(header_len_size)))
        header = f.read(header_len)
        if not header.endswith(b'\n'):
            raise ValueError('Invalid NPY file.')
        return f.tell()



def preprocessing_func(data_files):
    data_test = np.load(data_files[0])
    n = data_test.shape[0]
    dt = tf.as_dtype(data_test.dtype)
    return n, dt



def fixedlength_function(data_files):
    # npy_file = data_files[0]
    # header_offset = npy_header_offset(npy_file)

    # data_test = np.load(data_files[0])
    # NO_OF_FEATURES = data_test.shape[0]
    # dtype = tf.as_dtype(data_test.dtype)

    fixed_length_dataset = tf.data.FixedLengthRecordDataset(data_files,
                                                            NO_OF_FEATURES * dtype.size,
                                                            header_bytes= header_offset)
    return fixed_length_dataset


@tf.function
def extract_labels_files(file_path):
    label = tf.strings.split(file_path, os.sep)[-3]
    if label == 'healthy':
        label = 0
    elif 'used':
        label = 1

    dataset = tf.reshape(tf.io.decode_raw(file_path, dtype), (NO_OF_FEATURES, 1))
    return dataset, label



npy_file = files[0]
header_offset = npy_header_offset(npy_file)

NO_OF_FEATURES, dtype = preprocessing_func(files)

fixed_dataset = fixedlength_function(files)

ds = fixed_dataset

ds = ds.map(extract_labels_files, num_parallel_calls=tf.data.AUTOTUNE)

# creating batches of dataset
batch_size = 25

ds_train = ds.take(int(len(files))).batch(batch_size).repeat(10)
ds_test = ds.skip(int(len(files)*0.8)).batch(batch_size)

# total batches count of the dataset
total_batches = (int(len(files))/batch_size)


def model_call():
    no_of_epochs = 4
    model.fit(
        ds_train,
        epochs=no_of_epochs,
        validation_data=ds_test,
        steps_per_epoch=total_batches)

# for batch size 20
start = time.time()

model = Sequential()
model.add(layers.InputLayer(input_shape=(NO_OF_FEATURES)))
model.add(layers.Reshape((NO_OF_FEATURES, 1)))
# model.add(layers.Conv1D(4, 4, activation='relu'))
model.add(layers.MaxPool1D(pool_size=3000))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))


# print("length of training data: ",len(ds_train))
# print("length of testing data: ",len(ds_test))
print("Size per batch: ", batch_size)
print("Total number of batches: ", total_batches)

print(model.summary())
model.compile(loss=SparseCategoricalCrossentropy(),
              optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model_call()

end = time.time()
print("Total execution time: {t}sec".format(t=(round(end - start, 4))))
