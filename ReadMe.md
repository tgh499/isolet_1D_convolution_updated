Please go to https://dl.acm.org/doi/10.1145/3299815.3314429 for details of the feature embedding method. Once done, use the files in the following order.

How it works?

1. get a non-image dataset that satisfies the contraints of the above paper
2. generate distance_matrix using js_geodesic.py
3. generate t-SNE mappings for Jensen-Shannon (JS) using the distance matrix, and for Euclidean distance use the training set;
4. find the array representation using generate_tsne_mapped_dataset_ecl.py or generate_tsne_mapped_dataset_js.py
5. use the datasets with CNN; I used PyTorch; You may use TensorFlow. It doesn't matter.