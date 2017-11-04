from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from file_reader import FileReader
from Splitter import splits
from Tester import *

def predict(prod):

    print("Got inside Method")

    reader = FileReader("training.txt")
    X = reader.read_file()

    print("Starting SVD..")

    svd = TruncatedSVD(n_components=10, n_iter=10, random_state=42)
    dense = svd.fit_transform(X)

    print("Done with SVD, starting K Means...")

    km = KMeans(n_clusters=5, max_iter=100, n_init=1)
    ans = km.fit_predict(dense)
    centroids = svd.inverse_transform(km.cluster_centers_)

    index = reader.product[prod]
    cluster = km.predict(dense[index].reshape(1, -1))

    print("Done with K Means...")

    predictions = []
    for key in reader.product.keys():
        ind = reader.product[key]
        if ind < len(ans):
            if ans[ind] == cluster:
                predictions.append(key)

    return predictions

def learn():
    splits()
    print("Done Splitting...")

    tester = Tester()
    test_set = tester.getTestSet()

    print("Finished tester STuff")

    answers = []

    i = 0
    for prod in test_set:
        print("Inside Loop")
        answers.append(predict(prod))

        for f in range(1, 10):  # Documents how far along I am.
            if i >= f * len(test_set) // 10:
                print("Done with " + str(f) + "0% of learning...")
        i = i + 1

    print(tester.checkAnswers(answers))

if __name__== '__main__':
    learn()