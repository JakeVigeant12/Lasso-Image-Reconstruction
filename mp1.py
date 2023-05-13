from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import Lasso
from sklearn.model_selection import ShuffleSplit
from scipy.ndimage import median_filter

mpl.use('TkAgg')  # !IMPORTANT
BLOCK_SIZE = 16
IMAGE = "nature"
S = 10


def openImage(path, title):
    img = Image.open(path)
    bw_img = img.convert(mode="L")
    imgMatrix = np.asarray(bw_img).astype('float64')
    # plotImage(imgMatrix, title)
    print(imgMatrix.shape)
    return imgMatrix


def saveBlock(imageMatrix, k, startingCoordinates):
    endCoordinates = [(startingCoordinates[0] + k), (startingCoordinates[1] + k)]
    return imageMatrix[startingCoordinates[0]:endCoordinates[0], startingCoordinates[1]:endCoordinates[1]]


def corruptBlock(block, S):
    pixelNumbers = np.arange(0, block.size, 1)
    pixelNumbers = np.random.permutation(pixelNumbers)
    for i in range(block.size - S):
        pixelToCorrupt = pixelNumbers[i]
        pixelToCorruptX = (pixelToCorrupt) // (block[0].size)
        pixelToCorruptY = pixelToCorrupt % block[0].size
        block[pixelToCorruptX][pixelToCorruptY] = np.nan
    plotImage(block, IMAGE+str(S)+"BlockCorruption")
    plt.imshow(block)
    plt.show()
    return block


def plotImage(imgMatrix, title):
    plt.imshow(imgMatrix, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.savefig("./results/" + title + ".png")


def corruptImage(img, k, S):
    # loop through rows in sets of k
    # loop through the row slice in k's
    corrImg = np.zeros(np.shape(img))
    for i in range(0, len(img), k):
        for j in range(0, len(img[0]), k):
            currBlock = np.copy(saveBlock(img, k, [i, j]))
            currBlock = corruptBlock(currBlock, S)
            corrImg[i:i + k, j:j + k] = currBlock
    print("Corruption complete")
    return corrImg


def reconstructImage(img, k, S):
    # loop through corrupted image as before
    # grab each block, reconstruct it, replace it
    reconIm = np.zeros(np.shape(img))
    basisVectors = generate_2d_dct_basis_vectors(k, k)
    numBlocks = ((len(img) * len(img[0])) / (k * k))
    alphaImage = np.zeros((len(img) // k, len(img[0]) // k))
    blocksComplete = 0
    for i in range(0, len(img), k):
        for j in range(0, len(img[0]), k):
            currBlock = saveBlock(img, k, [i, j])
            recnBlock, alphaCurr = dct_lasso(currBlock, basisVectors)
            reconIm[i:i + k, j:j + k] = recnBlock
            blocksComplete += 1
            print(str((blocksComplete / numBlocks) * 100) + "% Complete with reconstruction of S = " + str(S))
            alphaImage[i // k][j // k] = np.log(alphaCurr)
            print("Alpha Added: " + str(np.log(alphaCurr)))
    min_value = np.min(alphaImage)
    max_value = np.max(alphaImage)
    pixelValues = (alphaImage - min_value) * (255 / (max_value - min_value))
    plotImage(pixelValues, "RegularizationPlot" + (IMAGE + str(S)))
    return reconIm


def generate_2d_dct_basis_vectors(P, Q):
    basis_vectors = np.zeros((P ** 2, Q ** 2))
    for x in range(1, P + 1):
        for y in range(1, Q + 1):
            for u in range(1, P + 1):
                for v in range(1, Q + 1):
                    if (u == 1):
                        alphu = (1 / P) ** 0.5
                    else:
                        alphu = (2 / P) ** 0.5
                    if (u == 1):
                        alphv = (1 / Q) ** 0.5
                    else:
                        alphv = (2 / Q) ** 0.5
                    vect = alphv * alphu * np.cos((np.pi * (2 * x - 1) * (u - 1)) / (2 * P)) * np.cos(
                        (np.pi * (2 * y - 1) * (v - 1)) / (2 * P))
                    basis_vectors[BLOCK_SIZE * (x - 1) + y - 1, BLOCK_SIZE * (u - 1) + v - 1] = vect

    return basis_vectors


# for MSE take MSE of sensed pixel value from test set compared to predicted value. well actually just end up
# throwing that out
def random_subset_cv(X, rasterImage, fullBasis, alphas, coords):
    n = X.shape[0]
    k = max(1, n // 6)
    cv = ShuffleSplit(n_splits=20, test_size=k, random_state=0)

    scores = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, fit_intercept=False)
        model_scores = []
        for train_index, test_index in cv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = rasterImage[train_index], rasterImage[test_index]
            # reconstruction
            lasso.fit(X_train, y_train)
            weights = lasso.coef_
            weights = weights.reshape(-1, 1)
            for idx in test_index:
                pixVal = rasterImage[idx]
                x = coords[idx][0]
                y = coords[idx][1]
                guess = np.dot(fullBasis[x * BLOCK_SIZE + y], weights)
                model_scores.append((pixVal - guess) ** 2)
        scores.append(np.mean(model_scores))
    optimal_alpha = alphas[np.argmin(scores)]
    plt.plot(np.log(alphas), scores)
    plt.xlabel("log10(alpha)")
    plt.ylabel("MSE")
    plt.grid(True)

    plt.savefig(IMAGE+str(S)+"CVcurve.png")
    plt.show()
    return optimal_alpha


def calcMSE(reconstruction, realImage):
    RSE = []
    reconstruction = reconstruction.reshape(-1, 1)
    realImage = realImage.reshape(-1, 1)
    for i in range(len(reconstruction)):
        RSE.append((reconstruction[i] - realImage[i]) ** 2)
    return np.mean(RSE)


def medianFilt(reconstruct):
    filtered_img = median_filter(reconstruct, size=3, mode='reflect')
    return filtered_img


def dct_lasso(image, full_basis_vectors):
    A = []
    rasterIm = image.reshape(-1, 1)
    # rasterize each basis vector as a column in the matrix, 64x64 matrix, each basis vector is 8x8, (64,1) x 64
    # recov = rasterIm.reshape(8,8)
    removedNana = []
    # remove nan pixels
    for pixel in range(len(rasterIm)):
        currPix = rasterIm[pixel]
        if (not (np.isnan(currPix))):
            removedNana.append(currPix)
    removedNana = np.array(removedNana)

    storeCoords = []
    # remove the basis vectors from the nan pixels
    for x in range(len(image)):
        for y in range(len(image[0])):
            pixVal = image[x][y]
            if not (np.isnan(pixVal)):
                storeCoords.append([x, y])
                A.append(full_basis_vectors[(x * BLOCK_SIZE) + y])

    A = np.array(A)
    alpha = random_subset_cv(A, removedNana, full_basis_vectors, np.logspace(-6, 6), storeCoords)

    lasso = Lasso(alpha=alpha, fit_intercept=False)

    lasso.fit(A, removedNana)
    weights = lasso.coef_
    weights = weights.reshape(-1, 1)
    remake = np.zeros(np.shape(image))
    # reconstruction
    for x in range(len(image)):
        for y in range(len(image[0])):
            pixVal = image[x][y]
            if ((np.isnan(pixVal))):
                pixVal = np.dot(full_basis_vectors[x * BLOCK_SIZE + y], weights)
            remake[x][y] = pixVal
    plotImage(remake, IMAGE+str(S)+"blockReconstruction")
    return remake, alpha


# CHANGE BEFORE NEXT SIM
# SlistNat = [150, 100, 50, 30, 10]
# SlistnormNat = np.array(SlistNat)/256
# MSENatnoFilt = [139.43978681999366, 243.89081775167475, 407.4012694977245, 553.1306421181037, 910.3360225682787]
# MSENatfilt = [253.8033276835361, 306.943217081178, 404.17449457977875, 510.2390442039506, 801.9298837370081]
# SlistBoat = [50, 40, 30, 20, 10]
# SlistnormBoat = np.array(SlistBoat)/64
# MSEBoatnoFilt = [32.15175144712543, 83.47316055918438, 166.58763852022102, 336.0610600287296, 758.7040606430869]
# MSEBoatfilt = [124.06081455207968, 150.36658022973583, 213.82697883787407, 325.12781862761466, 596.0976470002577]

#
# plt.plot(SlistnormNat, MSENatnoFilt, label='No Median Filter Nature')
# plt.plot(SlistnormNat, MSENatfilt, label='Median Filter Nature')
#
# plt.plot(SlistnormBoat, MSEBoatnoFilt, label='No Median Filter Boat')
# plt.plot(SlistnormBoat, MSEBoatfilt, label='Median Filter Boat')
# Add a legen
# plt.legend()
#
# # Add labels for the x and y axes
# plt.xlabel('S/K^2')
# plt.ylabel('MSE')
#
# # Add a title for the plot
# plt.title('Nature Image')
# plt.grid(True)
# # Show the plot
# plt.savefig("./results/SKvMSE")
# plt.show()
# def saveBlock(imageMatrix, k, startingCoordinates):
# for S in Slist:
realImg = openImage("./images/" + IMAGE + ".bmp", IMAGE)

# corrMat = corruptImage(realImg, BLOCK_SIZE, 10)
# plotImage(corrMat, (IMAGE + str(10) + "corrupted"))
# plotImage(realImg, IMAGE)
#     reconstruct = reconstructImage(corrMat, BLOCK_SIZE, S)
#     nofiltmse = calcMSE(reconstruct, realImg)
#     print("MSE no median filter: " + str(nofiltmse))
#     filtRecon = medianFilt(reconstruct)
#     filtmse = calcMSE(filtRecon, realImg)
#     print("MSE median filter: " + str(filtmse))
#     # S, no filt, filt
#     row = [S, nofiltmse, filtmse]
#     with open(IMAGE + '_MSE_values.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(row)
#     plotImage(reconstruct, (IMAGE + str(S) + "nofilt") + str(nofiltmse))
#     plotImage(filtRecon, (IMAGE + str(S) + "filt") + str(filtmse))

block = saveBlock(realImg, BLOCK_SIZE, [175, 150])
corrBlock = corruptBlock(block, S)
basis = generate_2d_dct_basis_vectors(BLOCK_SIZE, BLOCK_SIZE)
recon = dct_lasso(corrBlock, basis)
