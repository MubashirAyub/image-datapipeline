import time
import inspect
import sys
import tensorflow as tf
from DP_Folder.basemodel import Classification_Base_Model


def model_init(modelname, n_classes):

    """
    Selects and in instantiates wanted model based on name and number of classes.

    Parameters
    ----------
    modelname: str
        Name of the model. Must be exact in order for model selection to work
    n_classes: int
        Number of classes

    Returns
    -------
    model
        Instant of the wanted model.
    """
    classes = _get_classes()
    return classes[modelname](n_classes)


def _get_classes():
    """
    Returns all defined classes in a file. Used for model selection.

    Returns
    -------
    classes
        All defined classes in the file, this function is defined in.
    """
    classes = dict()
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            if name not in classes.keys():
                classes[name] = obj
    return classes

class MLP_Degree_Classifier(Classification_Base_Model):

    def __init__(self, n_classes):
        super().__init__(n_classes)

        # consistent parameters of the NN
        self.name = 'MLP_Degree_Classifier'
        self.rescale = tf.keras.layers.Rescaling(1./255)
        self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=2000, decay_rate=0.9)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.scheduler)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics = [tf.keras.metrics.CategoricalAccuracy()]
        self.cnn1 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(self.n_classes, activation=tf.nn.softmax)
        self.create_model()



    def create_model(self):
        inputs = self.rescale

        x = self.cnn1(inputs)
        x = self.maxpool1(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.dense3(x)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)


MODEL_LIST = list(_get_classes().keys())



# def model_training(train_ds, val_ds, number_of_classes, no_of_epochs):
#
#     start = time.time()
#     AUTOTUNE = tf.data.AUTOTUNE
#
#     train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#     val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
#     model = tf.keras.Sequential([
#       tf.keras.layers.Rescaling(1./255),
#       tf.keras.layers.Conv2D(64, 3, activation='relu'),
#       tf.keras.layers.MaxPooling2D(),
#       tf.keras.layers.Conv2D(32, 3, activation='relu'),
#       tf.keras.layers.MaxPooling2D(),
#       tf.keras.layers.Flatten(),
#       tf.keras.layers.Dense(32, activation='relu'),
#       tf.keras.layers.Dense(16, activation='relu'),
#       tf.keras.layers.Dense(number_of_classes)
#     ])
#     model.compile(
#       optimizer='adam',
#       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#       metrics=['accuracy'])
#
#     model.fit(
#       train_ds,
#       validation_data=val_ds,
#       epochs=no_of_epochs
#     )
#
#     end = time.time()
#     print("Total execution time: {t}sec".format(t=(round(end - start, 3))))
