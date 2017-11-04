from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from file_reader import read_file

# X = sparse_random_matrix(10, 10, density=0.5, random_state=42)

X = read_file("training-test.txt")

print(type(X))

svd = TruncatedSVD(n_components=10, n_iter=10, random_state=42)

dense = svd.fit_transform(X)

km = KMeans(n_clusters=5, max_iter=100, n_init=1)

ans = km.fit_transform(dense)

print(ans[0])