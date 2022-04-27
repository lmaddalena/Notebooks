import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import datasets, layers, models, optimizers
from datetime import datetime

def main():

    parser = argparse.ArgumentParser(description='Model parameters')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size', default=128)
    parser.add_argument('--epochs', dest='epochs', type=int, help='Number of epochs', default=20)

    args = parser.parse_args()

    # hyperparameters
    IMG_CHANNELS = 3
    IMG_ROWS = 32
    IMG_COLS = 32
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)
    CLASSES = 10
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIM = optimizers.RMSprop()    

    # load the dataset
    print("\nLoading dataset....")
    (X_train, Y_train), (X_test,Y_test) = load_dataset()

    # normalize the dataset
    print("\nnormalize the dataset....")    
    X_train = normalize_dataset(X_train)
    X_test = normalize_dataset(X_test)

    # one-hot representation of labels    
    Y_train = to_onehot(Y_train, CLASSES)
    Y_test = to_onehot(Y_test, CLASSES)

    # build the model    
    print("\nbuild the model....")    
    model = build_model((IMG_ROWS, IMG_COLS, IMG_CHANNELS), CLASSES)
    model.summary()

    # train the model
    print("\ntrain the model....")        
    history = train(
        model, 
        X_train, 
        Y_train, 
        OPTIM, 
        BATCH_SIZE, 
        EPOCHS, 
        VALIDATION_SPLIT, 
        VERBOSE)


    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    print('\n')    
    print(history_df.tail())

    # evaluate the model    
    print('\nEvalute the model:')
    score = model.evaluate(
        X_test, Y_test, 
        batch_size=BATCH_SIZE, 
        verbose=VERBOSE
    )    

    print("\nTest score: ", score[0])
    print("Test accuracy: ", score[1])    

def load_dataset():
    (X_train, Y_train), (X_test,Y_test) = datasets.cifar10.load_data()
    
    print("X_train: " + str(X_train.shape))
    print("Y_train: " + str(Y_train.shape))
    print("X_test: " + str(X_test.shape))
    print("Y_test: " + str(Y_test.shape))

    return (X_train, Y_train), (X_test,Y_test)

def normalize_dataset(X):    
    return X / 255.0

def to_onehot(Y, classes):
    Y = tf.keras.utils.to_categorical(Y, classes)
    return Y

def build_model(input_shape, classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu')) # fully connected
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation='softmax'))
    return model

def train(model, X, Y, optimizer, batch_size, epochs, validation_split, verbose):

    now = datetime.now()
    snow = now.strftime("%m/%d/%Y-%H:%M:%S")

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs/cifar10/scalars/' + snow)]

    model.compile(
        loss='categorical_crossentropy', 
        optimizer=optimizer, 
        metrics=['accuracy']
    )

    history = model.fit(
        X,
        Y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose
    )    

    return history

if __name__ == '__main__':
    main()