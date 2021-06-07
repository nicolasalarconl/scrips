# %%
import os

# %%
class AuxiliaryFunctions:
    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
