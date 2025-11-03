from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import load_iris
from moneyballer.preprocessor import X_proj
import pickle

# 1 - Load data (file path readable by the api app ; )
df = pd.read_csv("raw_data/FC26_20250921.csv")

#2 - Pre-Process data here onnly categorical data
# Preprocessing pipe : binarizing data + cleaning and preprocessing
cat_preproc = Pipeline([
    ("cat_imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preproc = ColumnTransformer([
        ("cat_tr", cat_preproc, make_column_selector(dtype_include=["object"]))
])

clean_preproc = Pipeline([
    ("cleaner", type_cleaner),
    ("preprocessing", preproc)])

pipe = Pipeline([
    ("clean_preproc", clean_preproc),
   pipe

#3 - Fit and transform and model description 
# Model description. Hist Gradiant Boosting improves the model’s performance by fitting each tree to the negative 
# gradient of the loss function with respect to the predicted value. RFs, on the other hand, are based on 
# bagging and use a majority vote to predict the outcome.

X = X.select_dtypes(include=['object'])
X.head()

X, y = load_iris(return_X_y=True)
clf = HistGradientBoostingClassifier().fit(X, y)
clf.score(X, y)


# save knn model as pickel file
with open("models/GradientBoosting.pkl", "wb") as file:
    pickle.dump(GradientBoosting_model, file)
    pickle.dump(GradientBoosting_model, file)