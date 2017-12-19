from sklearn.decomposition import TruncatedSVD
from file_reader import FileReader
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
    print(len(dense[0]))
    print(len(dense))
    print(len(svd.components_[0]))
    print(len(svd.components_))  # so anyways.. if I'm looking at product 4 and i want to see it's prediction... i need to find everything in that row... (wow) and then find the max
    # yeah so i'd need, so reduced[4][1] which is... 


    i = 0
    for prod in test_set:
        entry_in_matrix = reader.product[prod]

        row = list(copy.deepcopy(dense[entry_in_matrix]))
        ans = []
        for x in range(10):
            rec = np.argmax(row)
            ans.append(reader.ID[rec])
            row.remove(row[rec])

        answers.append(ans)
        if i % (len(test_set) // 100) == 0:
            print("\r Done with {}% of predicting...".format(i / len(test_set), end=''))
        i = i + 1


    print()
    print(tester.checkAnswers(answers))


if __name__== '__main__':
    learn()