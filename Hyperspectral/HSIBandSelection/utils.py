import cv2
import h5py
import torch
import pickle
from scipy import stats
from scipy import ndimage
from HSIBandSelection.readSAT import *
from scipy import integrate
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.metrics import r2_score
from dataclasses import dataclass, field
from numpy.random import binomial as binom
from sklearn.cross_decomposition import PLSRegression


np.random.seed(7)  # Initialize seed to get reproducible results
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def generic_gaussians(indices, bandwidth, L):
    """
    Given the indices of the wavelength centers and the
    filter bandwidth (given in nanometers), return the
    gaussians defined by these inputs
    """

    index_bandwidth = bandwidth
    all_gaussians = []

    # Given the bandwidth (which is equivalent to the full width-
    # half maximum of a Gaussian), calculate the standard deviation
    # FWHM = 2*sqrt(2ln(2))*stdev => stdev = FWHM/(2*sqrt(2ln(2)))
    stdev = np.divide(index_bandwidth, np.multiply(2, np.sqrt(np.multiply(np.log(2), 2))))

    for ind in indices:
        curve = np.linspace(0, L, L)
        pdf = norm.pdf(curve, ind, stdev)
        # pdf = np.multiply(pdf, np.divide(highest_count, np.max(pdf)))
        gauss = np.divide(pdf, np.max(pdf))  # Normalize to 1
        all_gaussians.append(gauss)
        # plt.plot(curve, pdf)

    return all_gaussians


def transform_data(produce_spectras, indices, bandwidth, L):
    """
    Get the original produce data, then transofrm that data
    using the histogram used to fit the histogram of wavelengths
    """

    gaussians = generic_gaussians(indices, bandwidth, L)

    new_data = []
    for spectrum in produce_spectras:
        old_spectrum = spectrum
        new_point = []

        # For each gaussian, integrate under the curve
        # multiplied by the original data point to
        # simulate a filter with the bandwidth of the
        # gaussian
        for g in gaussians:
            curve = np.multiply(old_spectrum, g)
            integral = integrate.trapz(curve)
            new_point.append(integral)

        # new_spectrum = np.multiply(old_spectrum, gaussian)

        new_data.append(new_point)

    return new_data


def add_rotation_flip(x, y):
    x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1))

    # Flip horizontally
    x_h = np.flip(x[:, :, :, :, :], 1)
    # Flip vertically
    x_v = np.flip(x[:, :, :, :, :], 2)
    # Flip horizontally and vertically
    x_hv = np.flip(x_h[:, :, :, :, :], 2)

    # Concatenate
    x = np.concatenate((x, x_hv, x_v))
    y = np.concatenate((y, y, y))

    return x, y


@dataclass
class Dataset:
    """Class used to store a dataset"""
    train_x: np.array
    train_y: np.array
    ind: list = field(default=None)
    name: str = 'temp'

    def __post_init__(self):
        if not self.ind:
            self.ind = [i for i in range(self.train_x.shape[-1])]


@dataclass
class Stats:
    """Class used to save performance statistics"""
    mean_accuracy: float
    std_accuracy: float
    mean_precision: float
    std_precision: float
    mean_recall: float
    std_recall: float
    mean_f1: float
    std_f1: float


def process_data(dataset: Dataset, flag_average=True, selection=None, transform=False, normalization=False):
    """Pre-process data before running band selection algorithms"""
    train_x, train_y, indexes = dataset.train_x, dataset.train_y, dataset.ind
    # Average consecutive bands
    if flag_average:
        img2 = np.zeros((train_x.shape[0], train_x.shape[1], train_x.shape[2], int(train_x.shape[3] / 2)))
        for n in range(0, train_x.shape[0]):
            for i in range(0, train_x.shape[3], 2):
                img2[n, :, :, int(i / 2)] = (train_x[n, :, :, i] + train_x[n, :, :, i + 1]) / 2.
        train_x = img2

    # Select indices
    if selection is not None:
        indexes = [i for i in range(train_x.shape[-1])]
        indexes.sort()
        print("Selecting bands: ", indexes)

    if transform:
        print("Transforming data...")
        nu = train_x.shape[0]
        w = train_x.shape[1]
        sp = train_x.shape[3]
        train_x = np.array(transform_data(produce_spectras=train_x.reshape((nu * w * w, sp)), bandwidth=5,
                                          indices=indexes, L=int(train_x.shape[3])))
        train_x = train_x.reshape((nu, w, w, len(indexes)))
    else:
        # Select bands from original image
        temp = np.zeros((train_x.shape[0], train_x.shape[1], train_x.shape[2], len(indexes)))
        for nb in range(0, len(indexes)):
            temp[:, :, :, nb] = train_x[:, :, :, indexes[nb]]
        train_x = temp.astype(np.float32)

    # Apply normalization to the entire dataset (used for the IBRA method)
    if normalization:
        train_x, _, _ = normalize(train_x)  # Later, during training, we will normalize each training set

    return Dataset(train_x, train_y, indexes, dataset.name)


def load_predefined_data(flag_average=True, median=False, nbands=np.inf, method='GSS', selection=None,
                         transform=False, data='', vifv=0, pca=False, pls=False, normalization=False, printInf=False):
    """Load one of the satellite HSI datasets"""
    compressed = False  # Flag used to load the compressed datasets
    if method == 'Compressed':
        compressed = True
    if data == "IP" or data == "PU" or data == "SA" or data == "KSC" or data == "BSW":
        train_x, train_y = loadata(data, compressed)
        train_x, train_y = createImageCubes(train_x, train_y, window=5)
    else:
        """Load Kochia or Avocado dataset"""
        hdf5_file = ''
        if data == "Kochia":
            hdf5_file = h5py.File('weed_dataset_w25.hdf5', "r")
        elif data == "Avocado":
            hdf5_file = h5py.File('avocado_dataset_w64.hdf5', "r")
        train_x = np.array(hdf5_file["train_img"][...]).astype(np.float32)
        train_y = np.array(hdf5_file["train_labels"][...])

    if printInf:
        print("Dataset shape: " + str(train_x.shape))
        print("Loading and transforming the data into the correct format...")
    # Average consecutive bands
    if flag_average:
        if data == 'Avocado':
            img2 = np.zeros(
                (train_x.shape[0], int(train_x.shape[1] / 2), int(train_x.shape[2] / 2), int(train_x.shape[3] / 2)))
        else:
            img2 = np.zeros((train_x.shape[0], train_x.shape[1], train_x.shape[2], int(train_x.shape[3] / 2)))
        for n in range(0, train_x.shape[0]):
            if data == 'Avocado':
                xt = cv2.resize(np.float32(train_x[n, :, :, :]), (32, 32), interpolation=cv2.INTER_CUBIC)
                for i in range(0, train_x.shape[3], 2):
                    if median:
                        img2[n, :, :, int(i / 2)] = (ndimage.median_filter(xt[:, :, i], size=5) +
                                                     ndimage.median_filter(xt[:, :, i + 1], size=5)) / 2.
                    else:
                        img2[n, :, :, int(i / 2)] = (xt[:, :, i] + xt[:, :, i + 1]) / 2.
            # Average consecutive bands
            else:
                for i in range(0, train_x.shape[3], 2):
                    if median and int(i / 2) > 120:
                        img2[n, :, :, int(i / 2)] = (ndimage.median_filter(train_x[n, :, :, i], size=7) +
                                                     ndimage.median_filter(train_x[n, :, :, i + 1], size=7)) / 2.
                    else:
                        img2[n, :, :, int(i / 2)] = (train_x[n, :, :, i] + train_x[n, :, :, i + 1]) / 2.

        train_x = img2

    # Select a subset of bands if the flag "selected is activated"
    
    if selection is not None:

        if selection is not None:
            indexes = [i for i in range(train_x.shape[3])]

        # Sort indexes
        indexes.sort()
        print("Selecting bands: ", indexes)
        print('2')

        if transform:
            print("Transforming data...")
            nu = train_x.shape[0]
            w = train_x.shape[1]
            sp = train_x.shape[3]
            train_x = np.array(transform_data(produce_spectras=train_x.reshape((nu * w * w, sp)), bandwidth=5,
                                              indices=indexes, L=int(train_x.shape[3])))
            train_x = train_x.reshape((nu, w, w, len(indexes)))
        else:
            # Select bands from original image
            temp = np.zeros((train_x.shape[0], train_x.shape[1], train_x.shape[2], len(indexes)))

            for nb in range(0, len(indexes)):
                temp[:, :, :, nb] = train_x[:, :, :, indexes[nb]]

            train_x = temp.astype(np.float32)

    # Apply normalization from the beginning (used for the IBRA method)
    if normalization:
        train_x, _, _ = normalize(train_x)

    return Dataset(train_x, train_y, indexes, data)


def normalize(trainx):
    """Normalize and returns the calculated means and stds for each band"""
    trainxn = trainx.copy()
    if trainx.ndim == 5:
        dim = trainx.shape[2]
    else:
        dim = trainx.shape[3]
    means = np.zeros((dim, 1))
    stds = np.zeros((dim, 1))
    for n in range(dim):
        if trainx.ndim == 5:  # Apply normalization to the data that is already in Pytorch format
            means[n,] = np.mean(trainxn[:, :, n, :, :])
            stds[n,] = np.std(trainxn[:, :, n, :, :])
            trainxn[:, :, n, :, :] = (trainxn[:, :, n, :, :] - means[n,]) / (stds[n,])
        elif trainx.ndim == 4:
            means[n,] = np.mean(trainxn[:, :, :, n])
            stds[n,] = np.std(trainxn[:, :, :, n])
            trainxn[:, :, :, n] = (trainxn[:, :, :, n] - means[n,]) / (stds[n,])
    return trainxn, means, stds


def applynormalize(testx, means, stds):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    for n in range(testx.shape[2]):
        testxn[:, :, n, :, :] = (testxn[:, :, n, :, :] - means[n,]) / (stds[n,])
    return testxn


def vif(inputX, printO=False, indexes=None):
    VIF = np.zeros((inputX.shape[3]))

    for i in range(0, inputX.shape[3]):
        y = inputX[:, :, :, i]
        x = np.zeros((inputX.shape[0], inputX.shape[1], inputX.shape[2], inputX.shape[3] - 1))
        c = 0
        for nb in range(0, inputX.shape[3]):
            if nb != i:
                x[:, :, :, c] = inputX[:, :, :, nb]
                c += 1
        x = x.reshape((inputX.shape[0] * inputX.shape[1] * inputX.shape[2], inputX.shape[3] - 1))
        y = y.reshape((inputX.shape[0] * inputX.shape[1] * inputX.shape[2], 1))
        model = sm.OLS(y, x)
        results = model.fit()
        rsq = results.rsquared
        VIF[i] = round(1 / (1 - rsq), 2)
        if printO:
            print("R Square value of {} band is {} keeping all other bands as features".format(
                indexes[i], (round(rsq, 2))))
            print("Variance Inflation Factor of {} band is {} \n".format(indexes[i], VIF[i]))

    return VIF


def tTest(method1='', nbands1=6, method2='', nbands2=6, data='', transform=False, file1=None, file2=None):
    """Perform a t-test between the results of two different methods."""

    if file1 is None and file2 is None:
        if transform:
            transform = 'GAUSS'
        else:
            transform = ''
        file1 = data + "\\results\\" + method1 + "\\" + str(nbands1) + " bands\\cvf1hyper3dnet" + method1 + \
                str(nbands1) + data + transform
        file2 = data + "\\results\\" + method2 + "\\" + str(nbands2) + " bands\\cvf1hyper3dnet" + method2 + \
                str(nbands2) + data + transform

    # Load the vectors of F1-scores obtain after 10-fold cross-validation
    with open(file1, 'rb') as f:
        cvf1 = pickle.load(f)

    with open(file2, 'rb') as f:
        cvf2 = pickle.load(f)

    for i in cvf1:
        print(i)
    print()
    for i in cvf2:
        print(i)

    return stats.ttest_rel(cvf1, cvf2).pvalue  # return the pvalue


def entropy(labels, base=2):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def get_class_distribution(train_y):
    """Get number of samples per class"""
    count_dict = {}

    # Iterate through the train_y and count occurrences of each class
    for i in train_y:
        if i not in count_dict:
            count_dict[i] = 1
        else:
            count_dict[i] += 1

    return count_dict


def getPCA(Xc, numComponents=5, dataset=None):
    """Reduce the number of components or channels using PCA"""
    newX = Xc.transpose((0, 3, 4, 2, 1))
    newX = np.reshape(newX, (-1, newX.shape[3]))
    pcaC = PCA(n_components=numComponents, whiten=True)
    newX = pcaC.fit_transform(newX)
    newX = np.reshape(newX, (Xc.shape[0], Xc.shape[3], Xc.shape[3], numComponents, Xc.shape[1]))
    newX = newX.transpose((0, 4, 3, 1, 2))
    # Save pca transformation
    if dataset is not None:
        file = dataset + "//results//PCA//pca_" + str(numComponents)
        with open(file, 'wb') as f:
            pickle.dump(pcaC, f)
    return newX, pcaC


def applyPCA(Xc, numComponents=5, transform=None, dataset='Kochia'):
    """Apply previously calculated PCA transformation"""
    if transform is not None:
        pcaC = transform
    else:
        # Load pca transformation
        file = dataset + "//results//PCA//pca_" + str(numComponents)
        with open(file, 'rb') as f:
            pcaC = pickle.load(f)
    newX = Xc.transpose((0, 3, 4, 2, 1))
    newX = np.reshape(newX, (-1, newX.shape[3]))
    print("Explained variance in the training set:")
    print(np.sum(pcaC.explained_variance_ratio_))
    print("Explained variance in the test set:")
    print(r2_score(newX, pcaC.inverse_transform(pcaC.transform(newX)), multioutput='variance_weighted'))
    newX = pcaC.transform(newX)
    newX = np.reshape(newX, (Xc.shape[0], Xc.shape[3], Xc.shape[3], numComponents, Xc.shape[1]))
    newX = newX.transpose((0, 4, 3, 1, 2))
    return newX


def getPLS(Xc, yc, numComponents=5, dataset=None):
    """Reduce the number of components or channels using PLS"""
    newX = Xc.transpose((0, 3, 4, 2, 1))
    newX = np.reshape(newX, (-1, newX.shape[3]))
    # Inflate the labels so that len(newY) matches len(newX)
    newY = []
    for yi in yc:
        for rep in range(int(len(newX) / len(yc))):
            newY.append(yi)
    PLS_transform = PLSRegression(n_components=numComponents)
    PLS_transform.fit(newX, newY)
    newX = PLS_transform.transform(newX)
    # print("Explained variance in the training set:")
    # print(np.sum(PLS_transform.explained_variance_ratio_))
    newX = np.reshape(newX, (Xc.shape[0], Xc.shape[3], Xc.shape[3], numComponents, Xc.shape[1]))
    newX = newX.transpose((0, 4, 3, 1, 2))
    # Save pls transformation
    if dataset is not None:
        file = dataset + "//results//PLS//pls_" + str(numComponents)
        with open(file, 'wb') as f:
            pickle.dump(PLS_transform, f)
    return newX, PLS_transform


def applyPLS(Xc, numComponents=5, transform=None, dataset='Kochia'):
    """Apply previously calculated PCA transformation"""
    if transform is not None:
        PLSC = transform
    else:
        # Load pca transformation
        file = dataset + "//results//PLS//pls_" + str(numComponents)
        with open(file, 'rb') as f:
            PLSC = pickle.load(f)
    newX = Xc.transpose((0, 3, 4, 2, 1))
    newX = np.reshape(newX, (-1, newX.shape[3]))
    newX = PLSC.transform(newX)
    newX = np.reshape(newX, (Xc.shape[0], Xc.shape[3], Xc.shape[3], numComponents, Xc.shape[1]))
    newX = newX.transpose((0, 4, 3, 1, 2))
    return newX


def permutationTest(scores1, scores2):
    """Perform a permutation paired t-test"""
    # Calculate the real differences between two samples
    d = np.array(scores1) - np.array(scores2)
    n = len(scores1)
    # Sets the number of iterations
    reps = 1000
    # Create permutation matrix
    x = 1 - 2 * binom(1, .5, 10 * reps)
    x.shape = (reps, n)
    # Apply permutations
    sim = x * d
    # Get distribution of simulated t-values
    dist = sim.mean(axis=1) / (sim.std(axis=1, ddof=1) / np.sqrt(n))
    # Get real t-value
    observed_ts = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    # Return 2-sided p-value
    return np.mean(np.abs(dist) >= np.abs(observed_ts))
