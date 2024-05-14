import datetime
import os
from pathlib import Path
import tensorflow as tf

import
import PPM_PressTemp.NN.template_datagenerator as data_gen
import PPM_PressTemp.NN.template_model as mdl

import pathlib
#from model import model_training


# def training_ds(ds_dir, val_split, subset, seed, img_height, img_width, batchsize):
#     train_ds = tf.keras.utils.image_dataset_from_directory(
#       ds_dir,
#       validation_split=val_split,
#       subset=subset,
#       seed=seed,
#       image_size=(img_height, img_width),
#       batch_size=batchsize)
#     return train_ds
#
#
# def validation_ds(ds_dir, val_split, subset, seed, img_height, img_width, batchsize):
#     val_ds = tf.keras.utils.image_dataset_from_directory(
#         ds_dir,
#         validation_split=val_split,
#         subset=subset,
#         seed=seed,
#         image_size=(img_height, img_width),
#         batch_size=batchsize)
#     return val_ds

def get_Dataloader(params):
    if 'data_set' in params:
        tr, vd = params['database'].load_datasets_from_json()
    else:
        if 'dist' in params and len(params['dist']) == 3:
            tr, vd = params['database'].get_datasets(dist=params['dist'])
        else:
            tr, vd = params['database'].get_datasets()
    print('tr: ', len(tr))
    print('val: ', len(vd))

    #print('tt: ', len(tt))

    if params['debug']:
       tr = tf.keras.utils.image_dataset_from_directory(
           params['database'],
           validation_split=0.2,
           subset="training",
           seed=123,
           image_size=(180, 180))
       vd = tf.keras.utils.image_dataset_from_directory(
           params['database'],
           validation_split= 0.2,
           subset="training",
           seed=123,
           image_size=(180, 180))
    else:
        training_generator = data_gen.datagenerator(tr, **params)
        validation_generator = data_gen.datagenerator(vd, **params)
        #test_generator = data_gen.datagenerator(tt, **params)

    return training_generator, validation_generator    #, test_generator


def train(modelname, trained_model=None, db=None, class_dict=None, log_dir='LOG_DIR', log_dir_info='', epochs=20, batch_size=5, workers=0,
          class_weight=None, debug=False, n_classes=None, **kwargs):
    if 'dataset_file' not in kwargs:
        dataset_file = None
    else:
        dataset_file = kwargs['dataset_file']

    # creating DataBaseHandler
    db_handler = DataBaseHandler(db=db, dataset_file=dataset_file)

    # Ref: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    # lists all available physical GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 4096 * 2)])  # Notice here
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    # generate class_dict if not provided
    if not class_dict:
        class_dict = db_handler.get_all_classes_in_db()
        print('Generated class_dict: ', class_dict)

    # calculating n_classes from class_dict if not provided
    if not n_classes:
        unique_classes = list()
        for val in class_dict.values():
            if val not in unique_classes:
                unique_classes.append(val)
        n_classes = len(unique_classes)

    # Loading pretrained model if specified
    params = {}
    if trained_model:
        print('\n### Loading Model')
        old_params = get_parameters(trained_model)
        params['loaded_from'] = trained_model
        model = mdl.model_init(old_params['model']['name'], old_params['model']['n_classes'])
        model.model = tf.keras.models.load_model(trained_model + '\model.h5')
    else:
        # creating new model
        model = mdl.model_init(modelname, n_classes)

    # changing parameters and path for debugging
    if debug:
        workers = 0
        batch_size = 1
        epochs = 10
        log_dir = os.path.join("../..", "model_logs",
                               "debug_{}_{}".format(model.name, datetime.datetime.now().strftime("%Y.%m.%d-%H%M%S")))
    else:
        # generating folder path for the model
        log_dir = os.path.join(log_dir,
                               "{}_{}_{}".format(datetime.datetime.now().strftime("%Y.%m.%d-%H%M%S"),
                                                 model.name, log_dir_info))

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # generating param dictionary
    params.update({'batch_size': batch_size, 'class_dict': class_dict, 'workers': workers, 'class_weight': class_weight,
                   'log_dir': log_dir, 'epochs': epochs, 'debug': debug, 'database': db_handler, 'database_path': db,
                   'model': model.model_parameters(), 'process_name': kwargs['process_name']})

    # add "dataset_file" to the params dict if specified

    if dataset_file:
        params['dataset_file'] = dataset_file
    if 'dist' in kwargs:
        params['dist'] = kwargs['dist']

    # calling the datagenerator generator to get training-, validation- and testdata
    training_generator, validation_generator, test_generator = get_Dataloader(params)

    # creating tensorboard logs for visualisation of training and validation
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=batch_size)

    # filewriter for the confusion matrix images
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')







if __name__ == '__main__':
    db = r'../.keras/datasets/flower_photos/'

    #dsfile = r'PATH TO DATASETFILE'  # not needed

    c_dict = None  # {'label_1': 0, 'label_2': 1,...} # only needed if differs from auto generated class_dict

    train(modelname='MLP_Degree_Classifier', process_name='', db=db, epochs=5, log_dir='runs',
          batch_size=64)




    # To fetch data directly after unzip the folder
    # archive = r'C:\Users\yc6jdse4\.keras\datasets\flower_photos.tgz'
    # data_dir1 = pathlib.Path(archive).with_suffix('')
    # print(data_dir1)


    #To fetch data directly from the directory (subfloders are labels)
    # data_dir = r"C:/Users/yc6jdse4/.keras/datasets/flower_photos/"
    # print(data_dir)
    #
    # batchSize = 16
    # img_h = 180
    # img_w = 180
    #
    # tr = training_ds(data_dir, 0.2, "training", 123, img_h, img_w, batchSize)
    # val = validation_ds(data_dir, 0.2, "validation", 123, img_h, img_w, batchSize)
    #
    # class_names_count = len(tr.class_names)
    # model_training(tr, val, class_names_count, 4)
