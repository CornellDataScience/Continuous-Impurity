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





    '''
    right_probs = f(X, params)
    left_probs = 1.0-right_probs
    left_sub = 0
    right_sub = 0
    labels = np.unique(y)
    for label in labels:
        right_probs_with_label = right_probs[np.where(y==label)]
        left_probs_with_label = left_probs[np.where(y==label)]
        right_sub += np.sum(right_probs_with_label)**2
        left_sub += np.sum(left_probs_with_label)**2
    right_probs_sum = np.sum(right_probs)
    left_probs_sum = np.sum(left_probs)

    if right_probs_sum == 0:
        left_sub_factor = 1.0/(X.shape[0] * left_probs_sum)
        left_sub *= left_sub_factor
        right_sub *= 1 - left_sub
    elif left_probs_sum == 0:
        right_sub_factor = 1.0/(X.shape[0] * right_probs_sum)
        right_sub *= right_sub_factor
        left_sub *= 1-right_sub_factor
    elif left_probs_sum == 0 and right_probs_sum == 0:
        print("SHOULDN'T HAPPEN")
    else:
        left_sub *= 1.0/(X.shape[0] * left_probs_sum)
        right_sub *= 1.0/(X.shape[0] * right_probs_sum)
    return 1 - left_sub - right_sub
    '''
