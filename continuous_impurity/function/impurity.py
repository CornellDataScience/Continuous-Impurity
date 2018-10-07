import numpy as np

def gini(groupings):
    sum = 0
    num_elems = 0
    for grouping in groupings:
        _, counts = np.unique(grouping, return_counts = True).astype(np.float32)
        fracs = counts/float(len(grouping))
        sum += np.sum(np.square(fracs))
        num_elems += len(grouping)
    sum /= float(num_elems)
    return 1 - sum

def expected_gini(subset_assign_probs, y):
    labels = np.unique(y)
    out = 1
    for k in range(subset_assign_probs.shape[1]):
        k_sum = 0
        for label in labels:
            kth_subset_probs_with_label = subset_assign_probs[np.where(y == label), k]
            k_sum += np.sum(kth_subset_probs_with_label)**2
        out -= k_sum/(y.shape[0] * np.sum(subset_assign_probs[:,k]))
    return out
