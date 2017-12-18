import random

from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.cluster import KMeans
from file_reader import FileReader
from Splitter import splits
from Tester import *
import numpy as np
import copy

def learn():

    reader = FileReader("recommendations-training.txt")
    X = reader.read_file()

    tester = Tester()
    test_set = tester.getTestSet()

    print("Finished splitting set into training set and testing set...")

    answers = []

    print("Starting SVD...")

    svd = TruncatedSVD(n_components=10, n_iter=10, random_state=42)
    dense = svd.fit_transform(X) # creates a dense, lower rank version of sparse matrix X.

    # Let's try doing this with out K means... Let's just try recommending with this somehow.

    i = 0
    for prod in test_set:
        entry_in_matrix = reader.product[prod]

        row = list(copy.deepcopy(dense[entry_in_matrix]))
        ans = []
        for x in range(10):
            rec = np.argmax(row)
            ans.append(reader.ID[rec])
            row.remove(row[rec])

        answers.append(ans)  # NOTE: I need to finish doing this with out K means.
        if i % (len(test_set) // 100) == 0:
            print("\rDone with {}% of predicting...".format(i / len(test_set)))
        i = i + 1


    print()
    print(tester.checkAnswers(answers))


if __name__== '__main__':
    learn()