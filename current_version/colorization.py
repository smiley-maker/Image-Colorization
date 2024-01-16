# Imports

# Network
from keras.layers import Conv2D, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from keras.models import Sequential
import tensorflow as tf
import numpy as np

# Images
import matplotlib.pyplot as plt
import cv2

# Utilities
import warnings
import os

warnings.filterwarnings("ignore")


class CNN():
    IMAGE_SIZE = (256, 256)
    TRAIN_SPLIT = 0.8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 200
    BATCH_SIZE = 32

    def __init__(self):
        print("processing data...")
        self.X, self.y = self.prepareInputData("../datasets/landscapes/")

        self.X_train = self.X[:int(len(self.X)*CNN.TRAIN_SPLIT)]
        self.X_test = self.X[int(len(self.X)*CNN.TRAIN_SPLIT):]
        self.y_train = self.y[:int(len(self.y)*CNN.TRAIN_SPLIT)]
        self.y_test = self.y[:int(len(self.y)*CNN.TRAIN_SPLIT)]

        print("creating colorization models...")
        self.model_hbl = self.unet()
        self.model_shc = self.unet()
        self.model_shl = self.unet()

        # self.model_hbl.compile(
        #     optimizer=Adam(learning_rate=CNN.LEARNING_RATE),
        #     loss=self.hue_bin_loss
        # )
        
        # self.model_shc.compile(
        #     optimizer=Adam(learning_rate=CNN.LEARNING_RATE), 
        #     loss=self.saturation_hue_loss
        # )

        # self.model_shl.compile(
        #     optimizer=Adam(learning_rate=CNN.LEARNING_RATE),
        #     loss=self.saturated_huber_loss
        # )

    def prepareInputData(self, path):
        X=[]
        y=[]
        for imageDir in os.listdir(path):
            try:
                img = cv2.imread(path + imageDir)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
                img = img.astype(np.float32)
                img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
                # resize the lightness channel to network input size 
                img_lab_rs = cv2.resize(img_lab, (CNN.IMAGE_SIZE[0], CNN.IMAGE_SIZE[1])) # resize image to network input size
                img_l = img_lab_rs[:,:,0] # pull out L channel
                img_ab = img_lab_rs[:,:,1:]#Extracting the ab channel
                img_ab = img_ab
                #The true color values range between -128 and 128. This is the default interval 
                #in the Lab color space. By dividing them by 128, they too fall within the -1 to 1 interval.
                X.append(img_l)
                y.append(img_ab)
            except:
                pass
        X = np.array(X)
        y = np.array(y)
        
        return X,y
    
    def unet(self):
        mod = Sequential()

        mod.add(Conv2D(16, (3,3), padding="same", strides=1, input_shape=(CNN.IMAGE_SIZE[0], CNN.IMAGE_SIZE[1], 1)))
        mod.add(LeakyReLU())
        mod.add(BatchNormalization())

        mod.add(Conv2D(32, (3,3), padding="same", strides=1))
        mod.add(LeakyReLU())
        mod.add(BatchNormalization())

        mod.add(Conv2D(64, (3,3), padding="same", strides=1))
        mod.add(LeakyReLU())
        mod.add(BatchNormalization())

        mod.add(Conv2D(32, (3,3), padding="same", strides=1))
        mod.add(LeakyReLU())
        mod.add(BatchNormalization())

        mod.add(Conv2D(16, (3,3), padding="same", strides=1))
        mod.add(LeakyReLU())
        mod.add(BatchNormalization())

        mod.add(Conv2D(2, (3,3), activation="tanh", padding="same", strides=1))

        return mod
    
    def hue_bin_loss(self, y_true, y_pred):
        a_true, b_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
        a_pred, b_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)

        hue_true = tf.math.atan2(b_true, a_true)
        hue_pred = tf.math.atan2(b_pred, a_pred)

        condition1 = tf.logical_and(tf.less(hue_true, 0), tf.less(hue_pred, 0))
        condition2 = tf.logical_and(tf.greater(hue_true, 0), tf.greater(hue_pred, 0))

        hl = tf.where(condition1 | condition2, 0.0, tf.abs(hue_pred - hue_true))

        saturation_true = tf.sqrt(tf.square(a_true) + tf.square(b_true))
        saturation_pred = tf.sqrt(tf.square(a_pred) + tf.square(b_pred))
        sl = tf.square(saturation_true - saturation_pred)        

        color_loss = tf.sqrt(tf.square(a_true - a_pred) + tf.square(b_true - b_pred))
        total_loss = tf.add(0.25*color_loss, (hl + 4*sl))

        return tf.reduce_mean(total_loss)  # Use reduce_mean to ensure a scalar loss value
    
    def saturated_huber_loss(self, y_true, y_pred):        
        a_true, b_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
        a_pred, b_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)

        condition1 = tf.logical_and(tf.less(y_true, 0), tf.less(y_pred, 0))
        condition2 = tf.logical_and(tf.greater(y_true, 0), tf.greater(y_pred, 0))

        hl = tf.where(condition1 | condition2, 0.0, tf.abs(y_pred - y_true))

        saturation_true = tf.sqrt(tf.square(a_true) + tf.square(b_true))
        saturation_pred = tf.sqrt(tf.square(a_pred) + tf.square(b_pred))
        sl = tf.square(saturation_true - saturation_pred)

        total_loss = hl + sl

        return tf.reduce_mean(total_loss)  # Use reduce_mean to ensure a scalar loss value
    
    def saturation_hue_loss(self, y_true, y_pred):
        # Split the predicted and ground truth tensors into a and b channels
        a_true, b_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
        a_pred, b_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
        
        # Compute the Euclidean distance between a and b channels 4
#        color_loss = tf.sqrt(tf.square(a_true - a_pred) + tf.square(b_true - b_pred))
        
        # Compute hue and saturation from a and b channels of predicted image
        hue_true = tf.math.atan2(b_true, a_true) #-pi to pi
        saturation_true = tf.sqrt(tf.square(a_true) + tf.square(b_true)) 
        hue_pred = tf.math.atan2(b_pred, a_pred)
        saturation_pred = tf.sqrt(tf.square(a_pred) + tf.square(b_pred))
        
        # Define the weighting factor for emphasizing saturation over hue
        saturation_weight = 2.0
        
        # Compute the weighted hue-saturation loss
        hue_saturation_loss = tf.square(hue_pred - hue_true) + saturation_weight * tf.square(saturation_pred - saturation_true)
        
        # Combine the color loss and hue-saturation loss
    #    total_loss = tf.add(0.5*color_loss,hue_saturation_loss)
        
        # Return the mean loss over the batch
        return tf.reduce_mean(tf.sqrt(hue_saturation_loss))
    
    def train(
            self
#            model : Sequential,
#            loss : hue_bin_loss or saturated_huber_loss or saturation_hue_loss

    ):
        
        print(f"setting up training loop...")

        optimizer_hbl = Adam(CNN.LEARNING_RATE)
        optimizer_shl = Adam(CNN.LEARNING_RATE)
        optimizer_shc = Adam(CNN.LEARNING_RATE)

        hbls = []
        shls = []
        shcs = []

        print(f"Epochs {CNN.NUM_EPOCHS}, steps per epoch {self.X_train.shape[0]//CNN.BATCH_SIZE}")

        for epoch in range(CNN.NUM_EPOCHS):
            for step in range(self.X_train.shape[0]//CNN.BATCH_SIZE):
                start_idx = CNN.BATCH_SIZE*step
                end_idx = CNN.BATCH_SIZE*(step+1)
                X_batch = self.X_train[start_idx:end_idx]
                y_batch = self.y_train[start_idx:end_idx]

                # Separate GradientTape for each model
                with tf.GradientTape() as tape_hbl, tf.GradientTape() as tape_shl, tf.GradientTape() as tape_shc:
                    y_prob_hbl = self.model_hbl(X_batch)
                    y_prob_shl = self.model_shl(X_batch)
                    y_prob_shc = self.model_shc(X_batch)

                    loss_hbl = self.hue_bin_loss(y_batch, y_prob_hbl)
                    hbls.append(loss_hbl)
                    loss_shl = self.saturated_huber_loss(y_batch, y_prob_shl)
                    shls.append(loss_shl)
                    loss_shc = self.saturation_hue_loss(y_batch, y_prob_shc)
                    shcs.append(loss_shc)

                    grads_hbl = tape_hbl.gradient(loss_hbl, self.model_hbl.trainable_variables)
                    optimizer_hbl.apply_gradients(zip(grads_hbl, self.model_hbl.trainable_variables))

                    grads_shl = tape_shl.gradient(loss_shl, self.model_shl.trainable_variables)
                    optimizer_shl.apply_gradients(zip(grads_shl, self.model_shl.trainable_variables))

                    grads_shc = tape_shc.gradient(loss_shc, self.model_shc.trainable_variables)
                    optimizer_shc.apply_gradients(zip(grads_shc, self.model_shc.trainable_variables))
                            
                            
                print(f"Epoch {epoch+1}/{CNN.NUM_EPOCHS} - Step {step+1}/{self.X_train.shape[0]//CNN.BATCH_SIZE} --- HBL Loss {tf.math.reduce_mean(hbls)} --- SHL Loss {tf.math.reduce_mean(shls)} --- SHC Loss {tf.math.reduce_mean(shcs)} ",end='\r' if step+1 < self.X_train.shape[0]//CNN.BATCH_SIZE else None)
        
        return self

    def get_random_image(self, folder_path):
        # Get a list of all files in the folder
        all_files = os.listdir(folder_path)

        # Filter only image files (you may need to adjust this based on your image file extensions)
        image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        # Check if there are any image files in the folder
        if not image_files:
            print("No image files found in the specified folder.")
            return None

        # Select a random image file
        random_image = np.random.choice(image_files)

        # Construct the full path to the selected image
        image_path = os.path.join(folder_path, random_image)

        return image_path
    
    def ExtractTestInput(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_ = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        img_=img.astype(np.float32)
        img_lab_rs = cv2.resize(img_, (CNN.IMAGE_SIZE[0], CNN.IMAGE_SIZE[1])) # resize image to network input size
        img_l = img_lab_rs[:,:,0] # pull out L channel
        img_l_reshaped = img_l.reshape(1,CNN.IMAGE_SIZE[0],CNN.IMAGE_SIZE[1],1)
        
        return img_l_reshaped

    def display_test(self, num_images):
        print("Gathering images for testing....")
        imgs = []
        available_datasets = ["../datasets/fruits/fruits/", "../datasets/pokemon/POKEMON/", "../datasets/pokemon1/images/images/", "../datasets/unsplash/train/train_data/", "../datasets/unsplash/test/test_data/"]
        for i in range(num_images):
            x = str(np.random.choice(available_datasets))
            img = self.get_random_image(x)
            imgs.append(img)

        # Create a single figure for all images
        plt.figure(figsize=(20, 5 * num_images))

        for idx, img in enumerate(imgs):
            img = cv2.imread(img)
            img = cv2.resize(img, (CNN.IMAGE_SIZE[0], CNN.IMAGE_SIZE[1]))

            # Extract test input
            image_for_test = self.ExtractTestInput(img)

            # Make predictions using the trained models
            model_hbl_pred = self.model_hbl(image_for_test)
            model_hbl_pred = model_hbl_pred * 128
            model_hbl_pred = model_hbl_pred.reshape(CNN.IMAGE_SIZE[0], CNN.IMAGE_SIZE[1], 2)

            model_shc_pred = self.model_shc.predict(image_for_test)
            model_shc_pred = model_shc_pred * 128
            model_shc_pred = model_shc_pred.reshape(CNN.IMAGE_SIZE[0], CNN.IMAGE_SIZE[1], 2)

            model_shl_pred = self.model_shl.predict(image_for_test)
            model_shl_pred = model_shl_pred * 128
            model_shl_pred = model_shl_pred.reshape(CNN.IMAGE_SIZE[0], CNN.IMAGE_SIZE[1], 2)

            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

            comp_hbl = img_lab.copy()
            comp_hbl[:, :, 1:] = model_hbl_pred
            comp_hbl = cv2.cvtColor(comp_hbl, cv2.COLOR_Lab2RGB)

            comp_shc = img_lab.copy()
            comp_shc[:, :, 1:] = model_shc_pred
            comp_shc = cv2.cvtColor(comp_shc, cv2.COLOR_Lab2RGB)

            comp_shl = img_lab.copy()
            comp_shl[:, :, 1:] = model_shl_pred
            comp_shl = cv2.cvtColor(comp_shl, cv2.COLOR_Lab2RGB)

            # Add subplots for each image
            plt.subplot(num_images, 4, idx * 4 + 1)
            img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_original)
            plt.title("original image")
            plt.axis('off')

            plt.subplot(num_images, 4, idx*4 + 2)
            plt.imshow(comp_hbl)
            plt.title('colorized with hbl')
            plt.axis('off')

            plt.subplot(num_images, 4, idx * 4 + 3)
            plt.imshow(comp_shc)
            plt.title("colorized with shc")
            plt.axis('off')

            plt.subplot(num_images, 4, idx * 4 + 4)
            plt.imshow(comp_shl)
            plt.title("colorized with shl")
            plt.axis('off')

        plt.show()
        plt.close()


if __name__ == "__main__":
    colorizer = CNN()

    colorizer.train()
#        model=colorizer.model_hbl,
#        loss=CNN.hue_bin_loss
#    )

    colorizer.display_test(10)