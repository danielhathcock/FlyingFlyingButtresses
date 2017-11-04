from __future__ import with_statement

from PIL import Image
import numpy as np
import os

class DataSet(object):


    def __init__(self, trainingPercent=.7, batchSize=100):
        self.trainingPercent = trainingPercent
        self.batchNum = 0
        self.batchSize = batchSize
        TRAIN_DIR = '/home/shyamal/Downloads/HomeDepot/image_classification/train/train/train'
        CATEGORIES_PATH = '/home/shyamal/Downloads/HomeDepot/train/Xy_train.txt'
        labels = {}
        labelDict = {'Chandeliers': 1, 'Showerheads': 2, 'Ceiling Fans': 3,
            'Vanity Lighting': 4, 'Floor Lamps': 5,
            'Single Handle Bathroom Sink Faucets': 6}
        with open(CATEGORIES_PATH, 'r') as f:
            for line in f:
                values = line.split('|')
                labels[values[0].strip()] = labelDict[values[1]]

        self.trainingData = np.empty((len(labels), 65, 65, 3), dtype=np.uint8)
        self.trainingLabels = np.empty((len(labels),), dtype=np.uint8)
        i = 0;
        for filename in os.listdir(TRAIN_DIR):
            if filename.endswith(".jpg"):
                filepath = os.path.join(TRAIN_DIR, filename)
                image = np.array(Image.open(filepath))
                self.trainingData[i] = image
                imageClass = labels[filename[:-4]]
                self.trainingLabels[i] = imageClass
                i += 1
                if (i % 1000 == 0) {
                    print(i)
                }
        permutation = np.arange(len(labels))
        np.random.shuffle(permutation)
        self.trainingData = self.trainingData[permutation]
        self.trainingLabels = self.trainingLabels[permutation]
    def nextBatch(self):
        trainingExamples = int(self.trainingPercent * len(self.trainingLabels))
        lower = (self.batchNum * self.batchSize) % trainingExamples
        self.batchNum += 1
        upper = (self.batchNum * self.batchSize) % trainingExamples
        i = lower
        batchImages = np.empty((self.batchSize, 65, 65, 3), dtype=np.uint8)
        batchLabels = np.empty((self.batchSize,), dtype=np.uint8)
        for j in range(self.batchSize):
            batchImages[j] = self.trainingData[i]
            batchLabels[j] = self.trainingLabels[i]
            i = (i + 1) % trainingExamples
        return batchImages, batchLabels


asdf = DataSet()
asdf.nextBatch()
