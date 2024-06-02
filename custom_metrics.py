import numpy as np
from sklearn import metrics
from scipy import stats
import warnings 


def MSE_score(targets, preds, mask):
    assert targets.shape == preds.shape, (targets.shape, mask.shape)
    values = []
    
    for i in range(len(targets)):
        m = mask[i]
        assert np.any(m), i
        t = targets[i][m]
        p = preds[i][m]
        if t.size == 0:
            values.append(0.0)
        else:
            values.append(np.sqrt(metrics.mean_squared_error(t, p)))
    
    return np.array(values)


def NSE_score(targets, preds, mask):
    assert targets.shape == preds.shape, (targets.shape, mask.shape)
    values = []
    
    for i in range(len(targets)):
        m = mask[i]
        t = targets[i][m]
        p = preds[i][m]
        
        if t.size == 0:
            values.append(1)
        else:
            values.append(metrics.r2_score(t, p)) 
    
    return np.array(values)


def pearson(targets, preds, mask):
    assert targets.shape == preds.shape, (targets.shape, preds.shape)
    values = []
    valid = 0
    for i in range(len(targets)):
        m = mask[i]
        t = targets[i][m]
        p = preds[i][m]
        
        if t.size == 0:
            values.append(1)
        else:
            with warnings.catch_warnings(record=True) as caught_list:
                v = stats.pearsonr(t, p)[0] 
                increment_valid = False
                for c in caught_list:
                    if isinstance(c.message, stats.ConstantInputWarning):
                        v = 0 
                        increment_valid = True
                if increment_valid: valid += 1
            values.append(float(v))
    
    values = np.array(values)
    # values = np.concatenate(values)
    
    return values


def smape_score(targets, preds, mask):
    # assert targets.ndim == preds.ndim == 2
    assert targets.shape == preds.shape, (targets.shape, mask.shape)
    values = []
    
    for i in range(len(targets)):
        m = mask[i]
        t = targets[i][m]
        p = preds[i][m]
        if t.size == 0:
            values.append(0)
        else:
            values.append(smape_helper(t, p))

    return np.array(values)


def smape_helper(targets, preds):
    nonzero = (targets !=0) & (preds !=0)
    t = targets[nonzero]
    p = preds[nonzero]
    
    value = np.sum(np.abs(t - p) / (np.abs(t) + np.abs(p)))
    return value / len(targets)


def CSI_score(targets, preds, mask, threshold=0.001):
    assert targets.shape == preds.shape, (targets.shape, mask.shape)
    values = []
    
    y_true = targets > threshold
    y_preds = preds > threshold 
    
    for i in range(len(targets)):
        m = mask[i]
        t = y_true[i][m]
        p = y_preds[i][m]
        if t.size == 0:
            values.append(1)
        else:
            
            tp = t * p 
            fp = (~t) * p 
            fn = t * (~p)
            
            tp = tp.sum() 
            fp = fp.sum()
            fn = fn.sum()
            
            csi = tp / (tp + fp + fn + 1e-8)
            
            values.append(csi)
    
    return np.array(values)

