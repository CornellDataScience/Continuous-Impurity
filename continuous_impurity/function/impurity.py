import numpy as np

def gini(groupings):
    out = 0
    num_elems = 0
    for grouping in groupings:
        counts = np.unique(grouping, return_counts = True)[1].astype(np.float32)
        out += len(grouping)*(1-np.sum(np.square(counts/float(len(grouping)))))
        num_elems += len(grouping)
    return out/float(num_elems)

def expected_gini(subset_assign_probs, y):
    labels = np.unique(y)
    out = 1
    for k in range(subset_assign_probs.shape[1]):
        k_sum = 0
        for label in labels:
            kth_subset_probs_with_label = subset_assign_probs[np.where(y == label), k]
            k_sum += np.sum(kth_subset_probs_with_label)**2
        #print("k_sum: ", k_sum)
        if k_sum != 0:
            out -= k_sum/(y.shape[0] * np.sum(subset_assign_probs[:,k]))
    return out
