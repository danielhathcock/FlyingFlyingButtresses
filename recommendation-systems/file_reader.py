import itertools
import collections
from scipy.sparse import csr_matrix

class FileReader:
    def __init__(self, path):
        text_file = open(path, "r")
        self.lines = text_file.read().split("\n")
        text_file.close()

        size = 0
        self.product = dict()  # Dictionary product is a dictionary : maps ID to integer

        # Map ID to integer
        for line in self.lines:
            words = line.split(",")
            for word in words:
                if word not in self.product.keys():
                    self.product[word] = size
                    size = size + 1

        self.real_data = collections.defaultdict(lambda: collections.defaultdict(int))  # POPULATING DATA

        for line in self.lines:
            words = line.split(",")
            pairwise = list(itertools.combinations(words, 2))  # with replacement?
            for pair in pairwise:
                self.real_data[self.product[pair[0]]][self.product[pair[1]]] = self.real_data[self.product[pair[0]]][self.product[pair[1]]] + 1


    def read_file(self, testing=True):
        data = list()  # from 1 to k
        row_ind = list()  # from 1 to k
        col_ind = list()  # from 1 to k

        for word_1 in self.product.keys():
            for word_2 in self.product.keys():
                if self.real_data[self.product[word_1]][self.product[word_2]] != 0:
                    row_ind.append(self.product[word_1])
                    col_ind.append(self.product[word_2])
                    data.append(self.real_data[self.product[word_1]][self.product[word_2]])

        X = csr_matrix((data, (row_ind, col_ind)))

        return X





