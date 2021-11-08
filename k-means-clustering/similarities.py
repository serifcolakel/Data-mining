# import tarfile
# import urllib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

datafile = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
labels_file = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"
#a
data = np.genfromtxt(
    datafile,
    delimiter=",",
    usecols=range(0, 4),
    skip_header=1
)
print(data)
#print(len(data))
true_label_names = np.genfromtxt(
    labels_file,
    delimiter=",",
    usecols=(1,),
    skip_header=1,
    dtype="str"
)
#s
#print("Data Type : ", true_label_names)

label_encoder = LabelEncoder()

true_labels = label_encoder.fit_transform(true_label_names)

# true_labels[:5]
#
# label_encoder.classes_


n_clusters = len(label_encoder.classes_)
preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
)

clusterer = Pipeline(
   [
       (
           "kmeans",
           KMeans(
               n_clusters=n_clusters,
               init="k-means++",
               n_init=50,
               max_iter=500,
               random_state=42,
           ),
       ),
   ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ]
)

pipe.fit(data)

preprocessed_data = pipe["preprocessor"].transform(data)

predicted_labels = pipe["clusterer"]["kmeans"].labels_

silhouette_score(preprocessed_data, predicted_labels)

adjusted_rand_score(true_labels, predicted_labels)

pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(data),
    columns=["Length", "Width"],
)

pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
pcadf["true_label"] = label_encoder.inverse_transform(true_labels)


plt.figure(figsize=(12, 18))

scat = sns.scatterplot(
    x="Length",
    y="Width",
    s=250,
    data=pcadf,
    hue="predicted_cluster",
    style="true_label",
    palette="Set2",
)

scat.set_title(
    "Iris Data Seti Icerisinde Benzerlikler"
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)
plt.show()