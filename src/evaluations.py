import os
import math
import copy
import warnings

import numpy as np
import scipy.sparse as sps
import scipy.stats as stats

from sklearn import metrics
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity

from src.TADs import tad_sim
from src.utils import create_directory
from src.visualizations import visualize_hic_contact_matrix

try:
	from scipy.stats import PearsonRConstantInputWarning
except:
	from scipy.stats import ConstantInputWarning as PearsonRConstantInputWarning



def mse(A:np.ndarray, B:np.ndarray):
    """evaluate the Mean Squared Error (MSE) between two scHi-C contact matrices

    Args:
        A (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix
        B (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix

    Returns:
        a scalar representing the MSE
    """
    if A.shape != B.shape:
        raise KeyError("matrices not of the same size")
    
    mse = np.square(np.subtract(A, B)).mean()
    if math.isnan(mse):
        raise ValueError("one or more input matrices are empty")
    
    return mse


def ssim(A:np.ndarray, B:np.ndarray, win_size:int=7):
    """evaluate the Structural Similarity Index Measure (SSIM) between two scHi-C contact matrices

    Args:
        A (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix
        B (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix
        win_size (int): an odd integer representing the window size for SSIM. Must be less than size of A and B

    Returns:
        a scalar representing the SSIM, normalized to remove negative values
    """
    if (len(A.shape) != 2) and (len(B.shape) != 2):
        raise ValueError("both input matrices are of the wrong shape")
    elif len(A.shape) != 2:
        raise ValueError("first input matrix is of the wrong shape (" + str(len(A.shape)) + " dimensions instead of 2)")
    elif len(B.shape) != 2:
        raise ValueError("second input matrix is of the wrong shape (" + str(len(B.shape)) + " dimensions instead of 2)")
    elif A.shape != B.shape:
        raise KeyError("matrices not of the same size")
    
    ssim = structural_similarity(A, B, multichannel=False, win_size=win_size, data_range=2)
    # ssim = tf.image.ssim(A, B, max_val=max(np.max(A), np.max(B)))
    if math.isnan(ssim):
        raise ValueError("one or more input matrices are empty")
    
    return ssim

def kendall_tau(A:np.ndarray, B:np.ndarray):
    """evaluate the Kendall's Tau Correlation between two scHi-C contact matrices

    Args:
        A (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix
        B (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix

    Returns:
        a tuple of scalars representing kendall's tau (normalized to remove negative values) and the p value
    """
    if (len(A.shape) != 2) and (len(B.shape) != 2):
        raise ValueError("both input matrices are of the wrong shape")
    elif len(A.shape) != 2:
        raise ValueError("first input matrix is of the wrong shape (" + str(len(A.shape)) + " dimensions instead of 2)")
    elif len(B.shape) != 2:
        raise ValueError("second input matrix is of the wrong shape (" + str(len(B.shape)) + " dimensions instead of 2)")
    elif A.shape != B.shape:
        raise KeyError("matrices not of the same size")
    
    tau, p_val = stats.kendalltau(A, B)
    if math.isnan(tau):
        raise ValueError("one or more input matrices are empty")
    
    return tau




######################################################### Hi-C Similarity Metrics ##########################################################################

def vstrans(d1, d2):
    """
    Variance stabilizing transformation to normalize read counts before computing
    stratum correlation. This normalizes counts so that different strata share similar
    dynamic ranges.
    Parameters
    ----------
    d1 : numpy.array of floats
        Diagonal of the first matrix.
    d2 : numpy.array of floats
        Diagonal of the second matrix.
    Returns
    -------
    r2k : numpy.array of floats
        Array of weights to use to normalize counts.
    """
    # Get ranks of counts in diagonal
    ranks_1 = np.argsort(d1) + 1
    ranks_2 = np.argsort(d2) + 1
    # Scale ranks betweeen 0 and 1
    nranks_1 = ranks_1 / max(ranks_1)
    nranks_2 = ranks_2 / max(ranks_2)
    nk = len(ranks_1)
    r2k = np.sqrt(np.var(nranks_1 / nk) * np.var(nranks_2 / nk))
    return r2k


def SCC(A:np.ndarray, B:np.ndarray, max_bins:int=40, correlation_method:str='PCC'):
    """
        Compute the stratum-adjusted correlation coefficient (SCC) between two
        Hi-C matrices up to max_dist. A Pearson correlation coefficient is computed
        for each diagonal in the range of 0 to max_dist and a weighted sum of those
        coefficients is returned.
        Parameters
        ----------
        mat1 : scipy.sparse.csr_matrix
            First matrix to compare.
        mat2 : scipy.sparse.csr_matrix
            Second matrix to compare.
        max_bins : int
            Maximum distance at which to consider, in bins.
        Returns
        -------
        scc : float
            Stratum adjusted correlation coefficient.
    """
    if (len(A.shape) != 2) and (len(B.shape) != 2):
        raise ValueError("both input matrices are of the wrong shape")
    elif len(A.shape) != 2:
        raise ValueError("first input matrix is of the wrong shape (" + str(len(A.shape)) + " dimensions instead of 2)")
    elif len(B.shape) != 2:
        raise ValueError("second input matrix is of the wrong shape (" + str(len(B.shape)) + " dimensions instead of 2)")
    elif A.shape != B.shape:
        raise KeyError("matrices not of the same size")
	
    if max_bins < 0 or max_bins > int(A.shape[0] - 5):
        max_bins = int(A.shape[0] - 5)


    mat1 = csr_matrix(A)
    mat2 = csr_matrix(B)
    
    corr_diag = np.zeros(len(range(max_bins)))
    weight_diag = corr_diag.copy()
    
    for d in range(max_bins):
        d1 = mat1.diagonal(d)
        d2 = mat2.diagonal(d)
        mask = (~np.isnan(d1)) & (~np.isnan(d2))
        d1 = d1[mask]
        d2 = d2[mask]
        # Silence NaN warnings: this happens for empty diagonals and will
        # not be used in the end.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=PearsonRConstantInputWarning
            )
            if correlation_method == 'PCC':
                cor = pearsonr(d1, d2)[0]

            elif correlation_method == 'SCC':
                cor = spearmanr(d1, d2)[0]
            else:
                print('Provided invalid correlation type')
                exit(1)
            corr_diag[d] = cor

        r2k = vstrans(d1, d2)
        weight_diag[d] = len(d1) * r2k

    corr_diag, weight_diag = corr_diag[1:], weight_diag[1:]
    mask = ~np.isnan(corr_diag)
    
    corr_diag, weight_diag = corr_diag[mask], weight_diag[mask]
    
    # Normalize weights
    weight_diag /= sum(weight_diag)
    
    # Weighted sum of coefficients to get SCCs
    scc = np.nansum(corr_diag * weight_diag)
    
    return scc


############################################################# GenomeDISCO ##############################################################################
# SRC https://github.com/kundajelab/genomedisco
def to_transition(mtogether):
    mtogether = csr_matrix(mtogether, dtype=np.float64)
    sums = np.asarray(mtogether.sum(axis=1)).flatten()
    sums = np.nan_to_num(sums, nan=1.0, posinf=1.0, neginf=1.0)
    sums[sums <= 1e-12] = 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        D = sps.spdiags(1.0 / sums, [0], mtogether.shape[0], mtogether.shape[1], format='csr')
    return D.dot(mtogether)


def random_walk(m_input, t):
    return m_input.__pow__(t)


def genome_disco(m1, m2, transition=True, tmax=3, tmin=3):
    np.fill_diagonal(m1, 0)
    np.fill_diagonal(m2, 0)
    
    m1 = csr_matrix(m1)
    m2 = csr_matrix(m2)
    
    # convert to an actual transition matrix
    if transition:
        m1 = to_transition(m1)
        m2 = to_transition(m2)

    # count nonzero nodes (note that we take the average number of nonzero nodes in the 2 datasets)
    rowsums_1 = m1.sum(axis=1)
    nonzero_1 = [i for i in range(rowsums_1.shape[0]) if rowsums_1[i] > 0.0]
    rowsums_2 = m2.sum(axis=1)
    nonzero_2 = [i for i in range(rowsums_2.shape[0]) if rowsums_2[i] > 0.0]
    nonzero_total = len(list(set(nonzero_1).union(set(nonzero_2))))
    nonzero_total = 0.5 * (1.0 * len(list(set(nonzero_1))) + 1.0 * len(list(set(nonzero_2))))

    scores = []
    if True:
        for t in range(1, tmax + 1):
            if t == 1:
                rw1 = copy.deepcopy(m1)
                rw2 = copy.deepcopy(m2)

            else:
                rw1 = rw1.dot(m1)
                rw2 = rw2.dot(m2)

            if t >= tmin:
                diff = abs(rw1 - rw2).sum()
                scores.append(1.0 * float(diff) / float(nonzero_total))
                
    # compute final score
    ts = range(tmin, tmax + 1)
    denom = len(ts) - 1
    if tmin == tmax:
        auc = scores[0]
        if 2 < auc:
            #auct = int(auc)
            auc = 2
        elif 0 <= auc <= 2:
            auc = auc
    else:
        auc = metrics.auc(range(len(ts)), scores) / denom

    reproducibility = (1 - auc)

    return reproducibility






def run_evals(generated, targets):
    scores = {
        'MSE':  [],
        'SSIM': [],
        'GD':   [], 
        'SCC':  [],
    }
    
    generated += 0.0000001
    
    for i in range(generated.shape[0]):
        scores['MSE'].append(mse(
            generated[i, 0, :, :].detach().to('cpu').numpy(),
            targets[i, 0, :, :].detach().to('cpu').numpy()
        ))
        scores['SSIM'].append(ssim(
            generated[i, 0, :, :].detach().to('cpu').numpy(),
            targets[i, 0, :, :].detach().to('cpu').numpy()
        ))
        scores['GD'].append(genome_disco(
            generated[i, 0, :, :].detach().to('cpu').numpy(),
            targets[i, 0, :, :].detach().to('cpu').numpy()
        ))
        scores['SCC'].append(SCC(
            generated[i, 0, :, :].detach().to('cpu').numpy(),
            targets[i, 0, :, :].detach().to('cpu').numpy()
        ))
        
        
    return scores


def run_all_evaluation(generated, targets, root, storage, PARAMETERS):
    visualizations = os.path.join(storage, 'visualizations')
    generated_visualizations = os.path.join(visualizations, 'generated')
    target_visualizations = os.path.join(visualizations, 'targets')

    create_directory(generated_visualizations)
    create_directory(target_visualizations)

    cooler_files = os.path.join(storage, 'cooler')
    create_directory(cooler_files)
    
    full_results_path = os.path.join(root, 'full_results.csv')
    write_header = not os.path.exists(full_results_path) or os.path.getsize(full_results_path) == 0
    with open(full_results_path, 'a+') as f:
        if write_header:
            f.write('tissue,stage,cell_type,cell_count,chromosome,start_bin,end_bin,SSIM,GenomeDISCO,SCC,TAD_Sim,MSE,KendallTau\n')
        for g, t in zip(generated, targets):
            file_identifier = g.split('/')[-1].split('.')[0]

            g = np.load(g)
            t = np.load(t)
            
            # Skip if matrices are empty or have no variance
            if g.size == 0 or t.size == 0 or np.std(g) < 1e-10 or np.std(t) < 1e-10:
                print(f"Skipping {file_identifier} - empty or constant matrix")
                continue
            
            # save the visualizations
            visualize_hic_contact_matrix(g, os.path.join(generated_visualizations, file_identifier + '.png'))
            visualize_hic_contact_matrix(t, os.path.join(target_visualizations, file_identifier + '.png'))
            
            # extract biological features
            try:
                f1_score = tad_sim(g, t, file_identifier, PARAMETERS, cooler_files)
            except Exception as e:
                print(f"TAD_Sim failed for {file_identifier}: {e}")
                f1_score = np.nan

            # Get all the parameters
            cell_counts = int(storage.split('/')[-1])
            cell_line = storage.split('/')[-2]
            tissue = storage.split('/')[-3]
            stage = storage.split('/')[-4]
            chromosome = file_identifier.split('_')[0]
            region_x = int(file_identifier.split('_')[1].split('s')[-1])
            region_y = int(file_identifier.split('_')[2].split('e')[-1])
            
            # Compute metrics with error handling
            try:
                ssim_val = ssim(g, t)
            except Exception:
                ssim_val = np.nan
            try:
                gd_val = genome_disco(g, t)
            except Exception:
                gd_val = np.nan
            try:
                scc_val = SCC(g, t)
            except Exception:
                scc_val = np.nan
            try:
                mse_val = mse(g, t)
            except Exception:
                mse_val = np.nan
            try:
                kt_val = kendall_tau(g, t)
            except Exception:
                kt_val = np.nan
            
            f.write(
                '{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                    tissue,
                    stage,
                    cell_line,
                    cell_counts,
                    chromosome,
                    region_x,
                    region_y,
                    ssim_val,
                    gd_val,
                    scc_val,
                    f1_score,
                    mse_val,
                    kt_val
                )
            )
            f.flush()
            
            print(tissue, stage, cell_line, cell_counts, chromosome, region_x, region_y, ssim_val, gd_val, scc_val, f1_score, flush=True)



        
    




def evaluate(results_path, PARAMETERS):
    stages = list(map(lambda x: os.path.join(results_path, x), os.listdir(results_path)))    
    stages = list(filter(lambda x: os.path.isdir(x), stages))
    
    for stage in stages:
        tissues = list(map(lambda x: os.path.join(stage, x), os.listdir(stage)))
        tissues = list(filter(lambda x: os.path.isdir(x), tissues))
        
        for tissue in tissues:
            cell_lines = list(map(lambda x: os.path.join(tissue, x), os.listdir(tissue)))
            cell_lines = list(filter(lambda x: os.path.isdir(x), cell_lines))
            
            for cell_line in cell_lines:
                cell_counts = list(map(lambda x: os.path.join(cell_line, x), os.listdir(cell_line)))
                cell_counts = list(filter(lambda x: os.path.isdir(x), cell_counts))
                
                for cell_count in cell_counts:
                    generated_folder = os.path.join(cell_count, 'generated')
                    targets_folder = os.path.join(cell_count, 'targets')

                    generated_files = sorted(list(map(lambda x: os.path.join(generated_folder, x), os.listdir(generated_folder))))
                    target_files = sorted(list(map(lambda x: os.path.join(targets_folder, x), os.listdir(targets_folder))))

                    run_all_evaluation(generated_files, target_files, results_path, cell_count, PARAMETERS)

