import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch
import torchvision.datasets as datasets
import torch.nn as nn

import tensorflow as tf
from tqdm import tqdm
from random import randint
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # Define a transform to pre-process the images
    train_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)
    batch_size = 50
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True, 
                              num_workers=4)
    # Define a model for extraction of the embeddings
    model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)

    embeddings = []
    embedding_size = 2048 # the actual embedding size from the picked model
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))

    # Use the model to extract the embeddings by: removeing the last layers of the model to access the embeddings the model generates. 
    model = nn.Sequential(*list(model.children())[:-1]) # remove last layer
    model.eval()
    with torch.no_grad():
      for i, image in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, _ = image
        outputs = torch.squeeze(model(inputs))
        embeddings[i*batch_size:((i+1)*batch_size)] = outputs
    np.save('dataset/embeddings.npy', embeddings)


def get_data():
    embeddings = np.load('dataset/embeddings.npy')
    train = pd.read_csv("train_triplets.txt", sep=" ", names=["A", "B", "C"])
    train["label"] = np.zeros(len(train), dtype= np.int8)

    for index in tqdm(range(len(train))):
        if randint(0, 1):
            temp = train.iloc[index, 1]
            train.iloc[index, 1] = train.iloc[index, 2]
            train.iloc[index, 2] = temp
            train.iloc[index, 3] = 1
    
    X_train = []
    for index in tqdm(range(len(train))):
        a = embeddings[train.iloc[index,0]]
        b = embeddings[train.iloc[index,1]]
        c = embeddings[train.iloc[index,2]]
        concated_features = list(np.concatenate((a, b, c), axis=0).astype(float))
        X_train.append(concated_features)
    
    scaler = MinMaxScaler()

    X_train = np.array(X_train)
    X_train = scaler.fit_transform(X_train)
    y_train = np.array(train["label"])

    test_data = pd.read_csv("test_triplets.txt", sep=" ", names=["A", "B", "C"])
    test = []
    for idx in tqdm(range(len(test_data))):
      a = embeddings[test_data.iloc[idx,0]]
      b = embeddings[test_data.iloc[idx,1]]
      c = embeddings[test_data.iloc[idx,2]]
      concated_features = list(np.concatenate((a, b, c), axis=0).astype(float))
      test.append(concated_features)
    test = np.array(test)

    test = scaler.transform(test)

    return train, X_train, y_train, test


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.5, input_shape=(6144,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1028, kernel_initializer="lecun_normal", activation="selu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, kernel_initializer="lecun_normal", activation="selu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, kernel_initializer="lecun_normal", activation="selu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, kernel_initializer="lecun_normal", activation="selu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="lecun_normal")
    ])
    model.summary()
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', 'mae'])
    return model


def train_model(X_train, y_train, train, model):
    def lr_scheduler(epoch, lr):
        if epoch < 10:
          return lr
        elif epoch < 15:
          lr = 1e-5
          return lr
        else:
          lr = 1e-6
          return lr 
      
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    model.fit(
      X_train[train.index], 
      y_train[train.index], 
      epochs=11, 
      batch_size=16,
      callbacks=[lr_schedule],
      verbose=1)
    return model
    

def test_model(model, test):
    y_test = model.predict(test)

    y_pred = []
    for i in range(len(y_test)):
      if(y_test[i][0] < 0.5):
        y_pred.append(1)
      else:
        y_pred.append(0)

    np.savetxt("results_project3.txt", np.array(y_pred), fmt='%d')


if __name__ == '__main__':
    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings.npy') == False): 
      generate_embeddings()

    # load the training and testing data
    train, X_train, y_train, test = get_data()

    # define model and train it
    model = create_model()
    model_trained = train_model(X_train, y_train, train, model)
    
    # test the model on the test data
    test_model(model, test)
    print("Results saved to results_project3.txt")
