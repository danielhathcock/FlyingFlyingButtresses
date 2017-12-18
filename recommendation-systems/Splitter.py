import numpy as np

# This method splits the training data and the test data.
def splits():
    f = open("recommendations-training.txt", 'r')
    lines = f.readlines()
    f.close()
    arr = np.arange(len(lines))
    np.random.shuffle(arr)
    f = open("training.txt", 'w')
    for i in range(4 * len(lines) // 5):
        f.write(lines[arr[i]])
    f.close()
    f = open("test.txt", 'w')
    for i in range(4 * len(lines) // 5, len(lines)):
        f.write(lines[arr[i]])
    f.close()
