# %%
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

# %%
print(tf.__version__)
print(keras.__version__)

# %%

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


# %%
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


# %%
# Setting up preprocessing images with InceptionV3
class PrepProcessImages:

    def load_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (299, 299))
        image = keras.applications.inception_V3(include_top=False, weights='imagenet')

        return image, path

    def apply_inceptionV3(self, ds_dir, train_image_names):
        from tqdm import tqdm

        img_model = tf.keras.applications.inception_v3(include_top=False, weights='imagenet')
        new_input = img_model.input
        hidden_layers = img_model.layers[-1].output
        img_features_extract_model = tf.keras.Model(new_input, hidden_layers)
        training_img_paths = [ds_dir + name + '.jpg' for name in train_image_names]
        encode_training_data = sorted(set(training_img_paths))
        img_ds = tf.data.Dataset.from_tensor_slices(encode_training_data)
        img_ds = img_ds.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(14)

        for img, path in tqdm(img_ds):
            batch_features = img_features_extract_model(img)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())


# %%
## Creating data_loader using LoadData() method

data_loader = LoadData()
img_processor = PrepProcessImages()

dataset_dir = "Flicker8k_Dataset/"
captions_file_path = "Flickr8k.token.txt"
train_img_file_path = "Flickr_8k.trainImages.txt"

captions_text = data_loader.load_text_doc(captions_file_path)
img_caps_dict = data_loader.image_caption_dict(captions_text)
training_img_names = data_loader.train_image_names(train_img_file_path)

img_processor.apply_inceptionV3(dataset_dir, training_img_names)

# %%
