import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    eps = 1e-9
    step = 1.0/n_bins
    bins = np.arange(0, 1 + eps, step)
    n = y_true.shape[0]
    ece = 0.0

    for st in bins :
        fin = st + step
        y_bin_true = y_true[((y_pred >= st) & (y_pred < fin))]
        y_bin_pred = y_pred[((y_pred >= st) & (y_pred < fin))]
        m = np.size(y_bin_true)
        if m == 0:
            continue 

        acc = np.mean(y_bin_true)
        conf = np.mean(y_bin_pred)
        ece += (m/n)*np.abs(acc - conf)

    return ece
        