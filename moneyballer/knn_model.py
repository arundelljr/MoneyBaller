from sklearn.neighbors import NearestNeighbors
from moneyballer.preprocessor import X_proj
import pickle

knn_model = NearestNeighbors(
    n_neighbors=101,        # 1 self + 100 similar players
    metric='cosine'       # best for similarity in high dimensions
)
knn_model.fit(X_proj)

# save knn model as pickel file
with open("models/knn_model.pkl", "wb") as file:
    pickle.dump(knn_model, file)
