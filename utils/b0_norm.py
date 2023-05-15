import numpy as np
import nibabel as nib


def B0normalize(newDataName):
    
    """

    Function that performs data normalization to the TE-specific B0, 
    and saves normalized data with filename newDataName.

    """

    # load data and scheme file
    data_nii = nib.load('/Users/theavage/Documents/Master/Data/GS35/P35_dwi.nii.gz')
    data = data_nii.get_fdata()
    scheme = np.loadtxt('/Users/theavage/Documents/Master/Data/GS35/GS35.scheme')

    ## perform the normalization

    # new volume on which to store data
    NormData = np.zeros(data.shape)

    # get the unique values of TE from the scheme
    TE_unique = np.unique(scheme[:, 6])

    # cycle between TE individual values
    for ii in range(len(TE_unique)):
        # get current TE
        cTE = TE_unique[ii]

        # compute the mean of the TE-specific B0
        B0s_idx = np.where((scheme[:, 3] == 0) & (scheme[:, 6] == cTE))[0]
        avg_B0 = np.mean(data[:, :, :, B0s_idx], axis=3, keepdims=True)

        # find TE-specific data indexes
        TE_idx = np.where(scheme[:, 6] == cTE)[0]

        # perform the normalization
        NormData[:, :, :, TE_idx] = data[:, :, :, TE_idx] / avg_B0

    ## store the normalized data into a new file

    # normalized data
    normNii = nib.Nifti1Image(NormData, data_nii.affine, data_nii.header)
    nib.save(normNii, newDataName)

