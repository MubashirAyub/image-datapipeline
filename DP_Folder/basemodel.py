import tensorflow as tf


class Classification_Base_Model:
    """
    Basemodel all new Models should inherit from.
    """

    def __init__(self, n_classes):
        self.name = 'Classification_Base_Model'
        self.n_classes = n_classes
        self.rescale = None
        self.mode = []
        self.model = tf.keras.Sequential()

        self.optimizer = None
        self.loss = None
        self.metrics = None

    def compile(self, summary=False):
        """ Compile Model

        Parameters
        ----------
        summary : bool
            Print summary if True.

        """
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)
        if summary:
            print(self.model.summary())

    def model_parameters(self):
        """ Calling this function will return current
            Parameters that are needed and set up for a model.

        Returns
        -------
        dict
            Parameters
        """
        params = {'n_classes': self.n_classes,
                  'name': self.name,
                  'mode': self.mode,
                  'rescale': self.rescale,
                  'metrics': [m.get_config() for m in self.metrics],
                  # 'metrics': self.metrics, #
                  'loss': self.loss.get_config(),
                  'optimizer': self.optimizer.get_config()
                  }
        return params

    def set_mode(self, mode):
        """ Set up modes of a model.
        Models are mainly used to differentiate between preprocessing variants.

        Parameters
        ----------
        mode : str
            The desired mode.

        """
        self.mode.append(mode)