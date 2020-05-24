#!/ddn/home4/r2444/anaconda3/bin/python
import pandas as pd
import numpy as np
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import copy
import math

def find_median_of_features(data_features):
    return(np.median(data_features, axis=0))

def generate_patches(image_sample, feature_medians):
    #image_sample_normalized = np.true_divide(image_sample, 255)
    for index, feature in enumerate(image_sample):
        if feature <= feature_medians[index]:
            image_sample[index] = 1
        else:
            image_sample[index] = 0

    greyIm = np.reshape(image_sample, (1,616))
    greyIm = greyIm[0]

    S=greyIm.shape
    N = 4
    patches = []
    #print(image_sample)
    count = 0
    max_index = len(greyIm)
    for col in range(max_index):
        count += 1
        Lx=np.max([0,col-N])
        Ux=np.min([max_index,col+N])
        patch = greyIm[Lx:Ux]
        if len(patch) == 8:
            patches.append(patch)
    return(patches)

def find_nearest_patch_universal_dict(sample_patches):
    encoded_sample = []
    for i in range(len(sample_patches)):
        patch_index = 0
        sample_patch = sample_patches[i]
        sample_patch = sample_patch.astype(int)
        sample_patch_str = ''.join(map(str, sample_patch))
        patch_index = int(sample_patch_str,2)
        encoded_sample.append(patch_index)

    return(encoded_sample)


def rearrange_feature_indices(perplexity, data_features):

    mapping_filename = "mapping_ecl_" + str(perplexity) + ".csv"
    mapping = pd.read_csv(mapping_filename, header=None)
    mapping = mapping.values

    if np.min(mapping) < 0:
        mapping = mapping - np.min(mapping)

    mapping_dict = {}

    for i,j in enumerate(mapping):
        mapping_dict[j[0]] = i

    mapping_keys_sorted = sorted(mapping_dict.keys())

    oneD_mapping = []
    for i in mapping_keys_sorted:
        oneD_mapping.append(mapping_dict[i])

    dim = 616

    tsne_mapped_data_features = np.zeros((len(data_features), dim))


    for i,j in enumerate(tsne_mapped_data_features):
        for k,l in enumerate(oneD_mapping):
            tsne_mapped_data_features[i][l] = data_features[i][k]

    return(tsne_mapped_data_features)


input_filename = 'test_randomized.csv'
data = pd.read_csv(input_filename, header=None)
data = data.head(1000)
features= data.columns[1:]
label = data.columns[0]
data_features = data[features]
data_label = data[label]
data_features_np = data_features.to_numpy()
data_label_np = data_label.to_numpy()
#codebook_patches_np = generate_sixteen_digit_codebook()

perplexities = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350]

for perplexity in perplexities:
    data_features_tsne_mapped = rearrange_feature_indices(perplexity, data_features_np)
    output_filename = "test_quantized_ecl_" + str(perplexity) + ".csv"

    feature_medians = find_median_of_features(data_features_tsne_mapped)
    new_dataset = []
    for i in range(len(data_features_tsne_mapped)):
        temp_new_sample = []
        sample_patches = generate_patches(data_features_tsne_mapped[i], feature_medians) # also gets converted to binary
        temp_new_sample.append(int(data_label_np[i]))
        temp_new_sample += find_nearest_patch_universal_dict(sample_patches)
        new_dataset.append(temp_new_sample)

    new_dataset_pd = pd.DataFrame(new_dataset)
    new_dataset_pd.to_csv(output_filename, encoding='utf-8', index=False, header=None)
