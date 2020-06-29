import numpy as np

def preprocess_lfs(
    L_train,
    L_mat,
    sim_from_mat_to_train
):
    '''
    Preprocess similarity scores and get the closest item in the support set for
    each LF.
    
    Args:
        L_train: The training matrix to look through to find nearest neighbors
        L_mat: The matrix to extend
        sim_from_mat_to_train: Similarity scores from L_mat to L_train.
            sim_from_mat_to_train[i][j] stores the similarity between element i of
            L_mat to element j of L_train.
            
    Returns:
        A tuple of three Numpy matrices.
        The first matrix stores which elements of L_mat have abstains,
        the second matrix stores, for each labeling function, the closest point in
        L_train where that same labeling function voted positive, and the third
        matrix stores, for each labeling function, the closest point in L_train
        where the labeling function voted negative.
    '''
    m = L_mat.shape[1]
    expanded_L_mat = np.copy(L_mat)

    train_support_pos = [
        np.argwhere(L_train[:, i] == 1).flatten()
        for i in range(m)
    ]
    train_support_neg = [
        np.argwhere(L_train[:, i] == -1).flatten()
        for i in range(m)
    ]

    mat_abstains = [
        np.argwhere(L_mat[:, i] == 0).flatten()
        for i in range(m)
    ]

    pos_dists = [
        sim_from_mat_to_train[mat_abstains[i]][:, train_support_pos[i]]
        for i in range(m)
    ]
    neg_dists = [
        sim_from_mat_to_train[mat_abstains[i]][:, train_support_neg[i]]
        for i in range(m)
    ]

    closest_pos = [
        np.max(pos_dists[i], axis=1)
        if pos_dists[i].shape[1] > 0 else np.full(mat_abstains[i].shape, -1)
        for i in range(m)
    ]
    closest_neg = [
        np.max(neg_dists[i], axis=1)
        if neg_dists[i].shape[1] > 0 else np.full(mat_abstains[i].shape, -1)
        for i in range(m)
    ]

    return mat_abstains, closest_pos, closest_neg

def extend_lfs(
    L_mat,
    mat_abstains,
    closest_pos,
    closest_neg,
    thresholds
):
    '''
    Extend LF's with fixed thresholds.
    
    Args:
        L_mat: The matrix to extend.
        mat_abstains, closest_pos, closest_neg: The outputs of the preprocess_lfs
            function.
        thresholds: The thresholds to extend each LF. For each item that an LF
            abstains on, if closest point that the LF votes on in the training
            is closer to the threshold, the LF is extended with the vote on that
            point. This information is encoded in mat_abstains, closest_pos, and
            closest_neg.
    
    Returns:
        An extended version of L_mat.
    '''
    m = L_mat.shape[1]
    expanded_L_mat = np.copy(L_mat)
    
    new_pos = [
        (closest_pos[i] > closest_neg[i]) & (closest_pos[i] > thresholds[i])
        for i in range(m)
    ]
    new_neg = [
        (closest_neg[i] > closest_pos[i]) & (closest_neg[i] > thresholds[i])
        for i in range(m)
    ]

    for i in range(m):
        expanded_L_mat[mat_abstains[i][new_pos[i]], i] = 1
        expanded_L_mat[mat_abstains[i][new_neg[i]], i] = -1
    
    return expanded_L_mat
