from nipype.interfaces import fsl
from nipype.interfaces.fsl import Merge
import nibabel as nib
from nipype.interfaces.fsl import EddyCorrect

eddy = EddyCorrect(in_file='VERDICT_all_b_values.nii.gz',out_file="diffusion_edc.nii.gz", ref_num=0)
print(eddy.cmdline)




