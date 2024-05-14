import numpy as np
from DP_Folder.basedatagenerator import datagenerator


class datagenerator(datagenerator):

    def load_obs(self, obs_path):
        # loading the data
        # ..todo:: Insert Code to load obs_path here
        return np.load(obs_path)