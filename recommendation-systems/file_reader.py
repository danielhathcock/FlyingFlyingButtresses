import itertools
import collections
from scipy.sparse import csr_matrix

def read_file(path, testing=True):
    text_file = open(path, "r")
    lines = text_file.read().split("\n")

    size = 0
    product = dict()  # Dictionary product is a dictionary : maps ID to integer

    # Map ID to integer
    for line in lines:
        words = line.split(",")
        for word in words:
            if word not in product.keys():
                product[word] = size
                size = size + 1

    real_data = collections.defaultdict(lambda: collections.defaultdict(int))  # POPULATING DATA

    for line in lines:
        words = line.split(",")
        pairwise = list(itertools.combinations(words, 2))  # with replacement?
        for pair in pairwise:
            real_data[product[pair[0]]][product[pair[1]]] = real_data[product[pair[0]]][product[pair[1]]] + 1

    data = list()  # from 1 to k
    row_ind = list()  # from 1 to k
    col_ind = list()  # from 1 to k

    for word_1 in product.keys():
        for word_2 in product.keys():
            if real_data[product[word_1]][product[word_2]] != 0:
                row_ind.append(product[word_1])
                col_ind.append(product[word_2])
                data.append(real_data[product[word_1]][product[word_2]])

    X = csr_matrix((data, (row_ind, col_ind)))

    print(X)

    return X





