#%%
import tensorflow as tf
from tensorflow import keras
import os

#%%
print(tf.__version__)
print(keras.__version__)

#%%

# Download text files
captions_folder = '/Flicker8k_text'
if not os.path.exists(os.path.abspath('.') + captions_folder):
    captions_zip = tf.keras.utils.get_file('Flicker8k_text.zip',
                                           cache_subdir=os.path.abspath('.'),
                                           origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
                                           extract=True)
    os.remove(captions_zip)

# Download image files
images_folder = '/Flicker8k_Dataset/'
if not os.path.exists(os.path.abspath('.') + images_folder):
    images_zip = tf.keras.utils.get_file('Flicker8k_Dataset.zip',
                                           cache_subdir=os.path.abspath('.'),
                                           origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
                                         extract=True)
    PATH = os.path.dirname(images_zip) + images_folder
    os.remove(images_zip)
else:
    PATH = os.path.abspath('.') + images_folder

#%%
class LoadData:
    def load_text_doc(self, file_path):
        with open(file_path, 'r') as cf:
            cap_text = cf.read()
        return cap_text

    def image_caption_dict(self, text):
        caption_mapping = {}

        lines = text.split('\n')
        for line in lines:
            line_split = line.split('\t')
            if len(line_split) < 2:
                continue
            else:
                image_meta, caption = line_split
                raw_image_name, caption_number = image_meta.split('#')
                image_name = raw_image_name.split('.')[0]

                if int(caption_number) == 0:
                    caption_mapping[image_name] = [caption]
                else:
                    caption_mapping[image_name].append(caption)

                return caption_mapping

    def train_image_names(self, file_path):
        data = []
        with open(file_path, 'r') as fp:
            text = fp.read()
            lines = text.split('\n')
            for line in lines:
                if len(line) < 1:
                    continue
                train_image_name = line.split('.')[0]
                data.append(train_image_name)
                return data

#%%
