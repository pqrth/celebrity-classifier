from vgg16 import vgg16

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.models import load_model
from scipy.misc import imread, imresize
import os
import numpy as np
import pickle
import datetime

import tensorflow as tf

import sys

iterations = 2
batch_size = 16
global datasetDir
datasetDir = "/home/scuser/IntelliJProjects/Python/ML-Training/celebClassifierDataset/"

def create_model(outputDim, inpDim = 4096):
    model = Sequential()
    model.add(Dense(outputDim * 256, input_dim=inpDim, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(outputDim * 16, input_dim=inpDim, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(outputDim, activation="softmax"))
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def train_model(model, outputDir, inputs, targets, v_inputs, v_targets):
    model.fit(np.array(inputs), targets, batch_size=batch_size, nb_epoch=iterations, validation_data = (np.array(v_inputs), v_targets))
    model.save(outputDir + "model.kmodel")
    model.save_weights(outputDir + "weights.h5")

def generateVocab():
    faces_dir = datasetDir + "face_images_orig/"
    files_dir = faces_dir
    local_vocab = []
    children = os.listdir(files_dir)
    for f in children:
        if os.path.isdir(os.path.join(files_dir, f)):
            local_vocab += [f]

    return local_vocab


def get_input_output():
    X = []
    Y = []
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    vocab = generateVocab()
    #print vocab

    faces_dir = datasetDir + "face_images_orig/"

    print "Loaded model...!"
    files_dir = faces_dir
    for subdir, dirs, filen in os.walk(files_dir):
        for f in filen:
            filename = os.path.join(subdir, f)
            name = subdir.split("/")[-1]
            print "Filename: " + filename
            try:
                image = imread(filename, mode='RGB')
                image = imresize(image, (224, 224))
            except Exception as e:
                print (e)
                continue
                
            probs, fc2s = sess.run([vgg.probs, vgg.fc2], feed_dict={vgg.imgs: [image]})
            X.append(fc2s[0])

            softmaxVector = [0] * len(vocab)
            softmaxVector[vocab.index(name)] = 1

            Y.append(softmaxVector)

    return X, Y

def get_input_output_fast():
    X = []
    Y = []
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    vocab = generateVocab()
    #print vocab

    faces_dir = datasetDir + "face_images_orig/"

    print "Loaded model...!"
    files_dir = faces_dir
    toProcessImages = []
    for subdir, dirs, filen in os.walk(files_dir):
        for f in filen:
            filename = os.path.join(subdir, f)
            name = subdir.split("/")[-1]
            print "Filename: " + filename
            try:
                image = imread(filename, mode='RGB')
                image = imresize(image, (224, 224))
            except Exception as e:
                print (e)
                continue
            toProcessImages.append(image)

            if (len(toProcessImages) > 25):
                print "Running a tensorflow session..."
                probs, fc2s = sess.run([vgg.probs, vgg.fc2], feed_dict={vgg.imgs: toProcessImages})
                for i in range (0, len(fc2s)):
                    X += [fc2s[i]]
                toProcessImages = []

            softmaxVector = [0] * len(vocab)
            softmaxVector[vocab.index(name)] = 1

            Y.append(softmaxVector)

    if (len(toProcessImages) > 0):
        probs, fc2s = sess.run([vgg.probs, vgg.fc2], feed_dict={vgg.imgs: toProcessImages})
        for i in range (0, len(fc2s)):
            X += [fc2s[i]]

    return X, Y

def train_with_inputs(outputDir):
    X, Y = get_input_output_fast()
    vocab = generateVocab()

    model = create_model(len(vocab), 4096)


    modelOutputDir = outputDir + "CelebClassifier_" + str(len(vocab)) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")+ "/"
    if not os.path.exists(modelOutputDir):
        os.makedirs(modelOutputDir)

    pickle.dump(vocab, open(modelOutputDir + "vocab.pkl", "wb"))

    print "Starting to train..."
    train_model(model, modelOutputDir, X, Y, X[-5:], Y[-5:])
    print "Training ended..."


def predict_celeb(model, sess, vgg, filename, vocab):
    image = imread(filename, mode='RGB')
    image = imresize(image, (224, 224))

    probs, fc2s = sess.run([vgg.probs, vgg.fc2], feed_dict={vgg.imgs: [image]})

    res = model.predict(np.array([fc2s[0]]))

    l = []
    for i in range(0, len(res[0])):
        l += [vocab[i], res[0][i]]

    print "Prediction for filename: " + filename
    # for i in range (len(l)):
    #     print l[i] + ": " + str(l[i])

    resString = ""
    for i in range (0, len(res[0])):
        resString += vocab[i] + ((": %0.4f") % res[0][i]) + "\n"

    print resString
    print "\n\n"


def doPredictions(modelDir):
    model = create_model(4096)

    model.load_weights(modelDir + "weights.h5")
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)


    print "loaded!"

    predict_celeb(model, sess, vgg, "/home/scuser/IntelliJProjects/Python/ML-Training/FaceDetection/amitabh1.jpg")
    predict_celeb(model, sess, vgg, "/home/scuser/IntelliJProjects/Python/ML-Training/FaceDetection/madhuri1.jpg")
    predict_celeb(model, sess, vgg, "/home/scuser/IntelliJProjects/Python/ML-Training/FaceDetection/sonam1.jpg")
    predict_celeb(model, sess, vgg, "/home/scuser/IntelliJProjects/Python/ML-Training/FaceDetection/sonam2.jpg")
    predict_celeb(model, sess, vgg, "/home/scuser/IntelliJProjects/Python/ML-Training/FaceDetection/emma1.jpg")
    predict_celeb(model, sess, vgg, "/home/scuser/IntelliJProjects/Python/ML-Training/FaceDetection/anushka1.jpg")


def doEvaluation(modelDir, outputDir=None):
    X, Y = get_input_output_fast()
    vocab = pickle.load(open(modelDir + "vocab.pkl", "rb"))
    model = create_model(len(vocab), 4096)
    model.load_weights(modelDir + "weights.h5")

    loss = model.evaluate(X, Y, batch_size=batch_size, verbose = 1)

    print "Total Loss: " + str(loss)


if __name__ == '__main__':
    if (len(sys.argv) < 1):
        print "At least call either train or predict. Aborting...\n"
        print "Training command style -> python CelebClassifier.py train outputDirPath datasetDirPath"
        exit(1)

    if (sys.argv[1] == "train"):
        datasetDir = sys.argv[3] + "/"
        train_with_inputs(sys.argv[2] + "/")
    elif (sys.argv[1] == "evaluate"):
        datasetDir = sys.argv[3] + "/"
        doEvaluation(sys.argv[2] + "/")
    else:
        doPredictions()
