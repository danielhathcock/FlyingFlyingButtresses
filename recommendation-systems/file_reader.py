import itertools
import collections
from Splitter import splits
from scipy.sparse import csr_matrix, dok_matrix

class FileReader:
    def __init__(self, path):
        text_file = open(path, "r")
        self.lines = text_file.read().strip().split("\n")
        text_file.close()

        print('Data fetched...')

        self.size = 0
        self.product = dict()  # Dictionary product is a dictionary : maps ID to integer

        # Map ID to integer
        for line in self.lines:
            for word in line.split(","):
                if word not in self.product:
                    self.product[word] = self.size
                    self.size += 1

        print('Assigned ids, there are {} products'.format(self.size))

        self.ID = {v : k for k,v in self.product.items()}

        self.real_data = collections.defaultdict(int)  # POPULATING DATA
        self.numNonzero = 0
        print(len(self.lines))

        splits() # Splits the data now.
        print("Done Splitting...")

        training_file = open("training.txt", "r")
        self.training_lines = training_file.read().strip().split("\n")

        i = 0
        maxVal = 0
        for line in self.training_lines:
            if i % (len(self.lines) // 10) == 0:
                print('\r{}/{}'.format(i, len(self.lines)), end='')
            words = line.split(",")

            pairwise = itertools.combinations_with_replacement(words, 2)  # with replacement? oh, so in each line, I have a line of products.
            for pair in pairwise:
                if self.real_data[ (self.product[pair[0]], self.product[pair[1]]) ] == 0:
                    self.numNonzero += 2
                self.real_data[ (self.product[pair[0]], self.product[pair[1]]) ] += 1
                maxVal = max(maxVal, self.real_data[ (self.product[pair[0]], self.product[pair[1]]) ])
            i += 1


        print(maxVal, self.numNonzero)


    def read_file(self, testing=True):

        sparse = dok_matrix((self.size, self.size)) # Question: How is the matrix set up? Products by Products?
        print('Created empty (sparse) matrix...')

        num = len(self.real_data)
        i = 0
        for a, b in self.real_data:
            if i % (num // 10) == 0:
                print('\r{}/{}'.format(i, num), end='')
            if self.real_data[(a, b)] == 0:
                print('rip?')
            else:
                sparse[a, b] = self.real_data[(a, b)]
            i += 1

        X = csr_matrix(sparse)
        print(X.nnz)

        return X




if __name__ == '__main__':
    f = FileReader('../product_recommendation/recommendations-training.txt')
    print('Got here')
    print(f.read_file())


