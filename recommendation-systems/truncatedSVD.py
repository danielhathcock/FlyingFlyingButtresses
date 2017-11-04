import random

from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.cluster import KMeans
from file_reader import FileReader
from Splitter import splits
from Tester import *

def predict(prod, products, ans, inverseAns):

    # print("Got inside Method")

    try:
        index = products[prod]
        # print('not rip')
    except KeyError:
        print('rip')
        return []

    cluster = ans[index]

    predictions = inverseAns[cluster]
    predictions = random.sample(predictions, min(len(predictions), 10))

    return predictions

def learn():
    # splits()
    print("Done Splitting...")

    tester = Tester()
    test_set = tester.getTestSet()

    print("Finished tester Stuff")

    answers = []

    reader = FileReader("training.txt")
    X = reader.read_file()

    print("Starting SVD..")

    svd = TruncatedSVD(n_components=10, n_iter=10, random_state=42)
    dense = svd.fit_transform(X)

    print("Done with SVD, starting K Means...")

    km = KMeans(n_clusters=100)
    ans = km.fit_predict(dense)

    print("Done with K Means...")

    inverseAns = {cluster: [] for cluster in range(100)}
    # centroids = svd.inverse_transform(km.cluster_centers_)
    for trainingProdKey, trainingProdIndex in reader.product.items():
        inverseAns[ans[trainingProdIndex]].append(trainingProdKey)

    print('Done inverting clusters')

    i = 0
    for prod in test_set:
        # print("Inside Loop")
        answers.append(predict(prod, reader.product, ans, inverseAns))

        if i % (len(test_set) // 100) == 0:
            print("\rDone with {}% of predicting...".format(i / len(test_set)), end='')
        i = i + 1

    print()
    print(tester.checkAnswers(answers))


if __name__== '__main__':
    learn()