from __future__ import with_statement

from PIL import Image
import numpy as np
import os

class DataSet(object):


    def __init__(self, trainingPercent=.7, batchSize=100):
        self.trainingPercent = trainingPercent
        self.batchNum = 0
        self.batchSize = batchSize
        TRAIN_DIR = './HomeDepot/ImagesTrain'
        CATEGORIES_PATH = './HomeDepot/categoriesTrain.txt'
        labels = {}
        labelDict = {'Plumbing': 1, 'Outdoors': 2, 'Flooring': 3,
            'Lighting & Ceiling Fans': 4, 'Appliances': 5}
        with open(CATEGORIES_PATH, 'r') as f:
            for line in f:
                values = line.split('|')
                labels[values[2].rstrip()] = labelDict[values[1]]

        self.trainingData = np.empty((len(labels), 65, 65, 3), dtype=np.uint8)
        self.trainingLabels = np.empty((len(labels),), dtype=np.uint8)
        i = 0;
        for part in os.listdir(TRAIN_DIR):
            if (part != '.DS_Store'):
                partPath = os.path.join(TRAIN_DIR, part)
                for filename in os.listdir(partPath):
                    if (partPath != '.DS_Store'):
                        filepath = os.path.join(partPath, filename)
                        if (labels[filename] != -1):
                            image = np.array(Image.open(filepath))
                            if (np.shape(image) == (65, 65)):
                                temp = np.empty((65, 65, 3), dtype=np.uint8)
                                temp[:,:,0] = temp[:,:,1] = temp[:,:,2] = image
                                image = temp
                            self.trainingData[i] = image
                            imageClass = labels[filename]
                            self.trainingLabels[i] = imageClass
                            labels[filename] = -1
                            i += 1
                            if (i % 10000 == 0):
                                print(i)
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
