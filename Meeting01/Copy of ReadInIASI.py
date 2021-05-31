import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def readIASIfun():
    import numpy as np
    import math
    import numpy.linalg as LA
    import matplotlib.pylab as plt
    import numpy.linalg as LA

    # Import the sample covariance matrix computed by the Desroziers diagnostic
    # Variables
    # Outputs:
    # sigma - diagonal matrix containing variances of each variable
    # Corr - full matrix containing the correlations of R
    # channellist - vector containing a list of the channels used
    # Rtype - type of matrix (Full, diagonal or band)
    # Cov - covariance matrix

    # Temporary variables:
    # i, k, lineincr, colentry, varentry, corentry, sigmalist

    # import necessary libraries

    fp = open('IASI_Rmatrix_Var_corr1_corr2_unpre', 'r')  # file where our data is stored\n",
    line = fp.readline()
    startline_raw = 0

    while line[
        0] == '!':  # skip over comments (lines starting with !) and store when the data starts in variable startline\n",
        line = fp.readline()
        startline_raw = startline_raw + 1
    ####read in set-up variables\n",

    # Satellite reference number\n",
    line = line.strip()
    columns = line.split()
    Satname_raw = int(columns[0])

    # Type of data we are reading in\n",
    line = fp.readline()
    line = line.strip()
    columns = line.split()
    Rtype_raw = int(columns[0])

    # Number of channels to read in\n",
    line = fp.readline()
    line = line.strip()
    columns = line.split()
    Channelnos_raw = int(columns[0])

    # Number of elements\n",
    line = fp.readline()
    line = line.strip()
    columns = line.split()
    Elementno_raw = int(columns[0])

    # Initialise our vectors and matrices based on the extracted parameters\n",
    sigmalist_raw = np.zeros(
        Channelnos_raw)  # set up vector where variances will be stored - this will then be converted to a diagonal matrix\n",
    Corr = np.zeros((Channelnos_raw, Channelnos_raw))  # set up correlation matrix\n",
    channellist_raw = np.zeros(Channelnos_raw)  # set up list of used channels\n",
    lineincr_raw = int(np.floor(Channelnos_raw / 10))  # number of full lines of 10 per 'vector'\n",
    linerem_raw = int(Channelnos_raw) - 10 * lineincr_raw  # number of entries in final line of each 'vector'\n",

    # Vector containing channel list\n",
    for k in range(0, lineincr_raw):  # read in all full rows\n",
        line = fp.readline()
        line = line.strip()
        columns = line.split()
        for j in range(0, 10):  # Extract entries from each row\n",
            colentry_raw = float(columns[j])
            index1_raw = 10 * (k) + j;
            channellist_raw[index1_raw] = colentry_raw
    # final spillover row if needed\n",
    if linerem_raw != 0:
        line = fp.readline()
        line = line.strip()
        columns = line.split()
        for j in range(0, linerem_raw):  # Extract entries from final row\n",
            colentry_raw = int(columns[j])
            index1_raw = 10 * (lineincr_raw) + j;
            channellist_raw[index1_raw] = colentry_raw

    # Vector containing variances - will be converted to diagonal matrix    \n",
    for k in range(0, lineincr_raw):  # read in all full rows\n",
        line = fp.readline()
        line = line.strip()
        columns = line.split()
        for j in range(0, 10):  # extract entries from each row and store in sigmalist\n",
            varentry_raw = float(columns[j])
            index1_raw = 10 * (k) + j;
            sigmalist_raw[index1_raw] = varentry_raw

    # Final spillover row if needed\n",
    if linerem_raw != 0:
        line = fp.readline()
        line = line.strip()
        columns = line.split()
        for j in range(0, linerem_raw):  # extract entries from each row and store in sigmalist\n",
            varentry_raw = float(columns[j])
            index1_raw = 10 * (lineincr_raw) + j;
            sigmalist_raw[index1_raw] = varentry_raw
    # read variance vector into diagonal \n",
    sigma_raw = np.diag(sigmalist_raw)

    # cycle over the block entry corresponding to each matrix row\n",
    for m_raw in range(0, int(Channelnos_raw)):

        for k in range(0, lineincr_raw):  # First 33 lines of each block\n",
            line = fp.readline()
            line = line.strip()
            columns = line.split()
            for j in range(0, 10):  # extract values into correct matrix entry\n",
                corentry_raw = float(columns[j])
                index1_raw = 10 * (k) + j  # determine position in matrix where this entry should be added\n",
                # print index1\n",
                Corr[m_raw, index1_raw] = corentry_raw
        if linerem_raw != 0:
            line = fp.readline()
            line = line.strip()
            columns = line.split()
            for j in range(0, linerem_raw):
                corentry_raw = float(columns[j])
                index1_raw = 10 * (k + 1) + j
                Corr[m_raw, index1_raw] = corentry_raw
    fp.close()

    # Compute covariance matrix from correlation and standard deviations
    Cov = np.dot(sigma_raw, np.dot(Corr, sigma_raw.T))

    # May also be interested only in correlated channels
    IASIcorrvec = np.array([1, 3, 5, 8, 26, 29, 31, 33, 36, 38, 41, 43, 46, 48, 50, 52, 53, 54,
                            55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                            73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92,
                            93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                            111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 129,
                            130, 131, 133, 134, 135, 136, 138, 139, 142, 144, 145, 162, 163, 164, 166, 169, 170, 175,
                            177, 178, 182, 183, 184, 185, 188, 194, 195, 199, 200, 201, 214, 220, 250, 258, 260, 262,
                            269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279]);

    CovSubtemp = Cov[IASIcorrvec]  # copy the correct columns into a temporary array
    CovSub = CovSubtemp[:, IASIcorrvec]  #

    CorrSubtemp = Corr[IASIcorrvec]
    CorrSub = CorrSubtemp[:, IASIcorrvec]
    sigmaSub = np.diag(sigma_raw[IASIcorrvec, IASIcorrvec])

    return CorrSub, CovSub, sigmaSub


CorrSub, CovSub, sigmaSub = readIASIfun()

df_cm = pd.DataFrame(CorrSub, range(137), range(137))
# plt.figure(figsize=(10,7))
corr = df_cm.corr()
ax = sn.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sn.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

plt.show()
