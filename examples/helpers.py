from flyingsquid.label_model import LabelModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

def train_fs_model_spam(L_train):
    label_model = LabelModel(L_train.shape[1])
    
    label_model.fit(
        L_train,
        
        # These parameters tuned for spam
        class_balance = np.array([0.4, 0.6]),
        solve_method = 'triplet_median'
    )
    
    return label_model

def evaluate_fs_model_spam(label_model, L_mat, Y_mat):
    preds = label_model.predict_proba_marginalized(L_mat)
    
    preds = [ 1 if pred > 0.5 else -1 for pred in preds ]
    
    pre, rec, f1, support = precision_recall_fscore_support(
        Y_mat, preds
    )
    acc = accuracy_score(Y_mat, preds)
    
    print('Acc: {:.4f}\tPre: {:.4f}\tRec: {:.4f}\tF1: {:.4f}'.format(
        acc, pre[1], rec[1], f1[1] 
    ))