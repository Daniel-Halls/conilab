import pandas as pd
import nilearn.image as img
from nilearn.datasets import load_mni152_template
import scipy.stats as stats
from nilearn.maskers import NiftiMasker
import glob
import re
import argparse
import warnings
warnings.filterwarnings("ignore")

def args() -> dict:
    '''
    Function to get arguments for correlations

    Parameters
    ----------
    None

    Returns
    -------
    dict: dictionary
       dictionary of args
    '''
    option = argparse.ArgumentParser()
    option.add_argument('-C', '--correlation_type',
                        dest='correlation_type',
                        help="""What type of correlation to run. 
                        Either just correlates from masks or does whole brain
                        put as 'whole' or 'mask' """,
                        type=str)
    option.add_argument('-i', '--image',
                        dest='image',
                        help='Abosulte path to image',
                        required=True,
                        type=str)
    option.add_argument('-n', '--number_of_components', 
                        dest='number_of_components',
                        help='Number of components',
                        required=True,
                        type=int)
    option.add_argument('-a', '--atlas', 
                        dest='atlas', 
                        help='absolute path to where atlas of Xtract is',
                        required=True,
                        type=str)
    option.add_argument('-c', '--csv',
                        dest='csv',
                        help='Abosulte path to where to save csv',
                        required=True,
                        type=str)
    option.add_argument('-t', '--threshold',
                        dest='threshold',
                        help='Whether to threshold the image components before running correlations and set threshold',
                        type=float)
    option.add_argument('--csv_name',
                        dest='csv_name',
                        help='What to name the csv. Default is correlations_to_tracts or correlations_to_tracts_thresholded',
                        type=str)
    return vars(option.parse_args())

def correlation_type(corr_type: str) -> str:
    """
    Function to determine correlation type
    
    Parameters
    ----------
    corr_type: str
        string of correlation type
    
    Returns
    -------
    str: string
        string of correlation type
    """
    avail_type = ['whole', 'mask']    
    if not corr_type:
        print(f'no -C given. Doing whole brain correlation')
        return avail_type[0]
    if corr_type.lower() in avail_type:
        print(f'Doing {corr_type.lower()} correlation')
        return corr_type.lower()
    
    print(f'{corr_type.lower()} not avaiable as correlation type. Doing whole brain correlation')
    return avail_type[0]
   
   
        
def mask_correlation(tract_mask, comp_img, tract_img) -> dict:
    """
    Fuction to extract signal from an 
    anatomical mask.

    Parameters
    ----------
    tract_mask: NIFIT img
        binarized mask of tract
    comp_img: NIFIT image
        component nifit
    tract_img: NIFIT image
        tract image
    """
    nifti_masker = NiftiMasker(
        mask_img=tract_mask,
        standardize=False, 
        memory="nilearn_cache")
    return {
        'component': nifti_masker.fit_transform(comp_img),
        'tract': nifti_masker.fit_transform(tract_img)
    }


def whole_brain_correlation(comp_img, tract_img) -> dict:
    """
    Function to return whole brain correlation

    Parameters
    ----------
    comp_img: NIFIT image
        component nifit
    tract_img: NIFIT image
        tract image
    """
    return {
        'component': comp_img.get_fdata().ravel(),
        'tract': tract_img.get_fdata().ravel()
    }

if __name__=='__main__':
    options = args()
    nfm = img.load_img(options['image'])
    corr_type = correlation_type(options['correlation_type'])

    if corr_type == 'whole':
        template = load_mni152_template(resolution=1)
    
    atlas = glob.glob(f"{options['atlas']}/*.nii.gz")
    pattern = re.compile(r'/(?P<word>[^/]+)\.nii\.gz')
    key = [match.group('word') for path in atlas if (match := pattern.search(path))]
    n_components = options['number_of_components']
    correlation_dict = dict(zip(key, [[] for at in key]))
    
    for tract_img in atlas:
        tract = img.load_img(tract_img)
        if corr_type == 'whole':
            resample_tract = img.resample_to_img(tract, template, interpolation='nearest')
        if corr_type == 'mask':
            tract_mask = img.binarize_img(tract)

        name = re.findall(r'/(?P<word>[^/]+)\.nii\.gz', tract_img)[0]
        print(f'Working on {name}')
        for component in range(0, n_components):
            print(component, end='\r')
            comp = img.index_img(nfm, component)
            if options['threshold']:
                    comp = img.threshold_img(comp, threshold=options['threshold'])
            if corr_type == 'whole':
                resample_component = img.resample_to_img(comp, template, interpolation='nearest')
                corr_xy =  whole_brain_correlation(resample_component, resample_tract)
            if corr_type == 'mask':
                corr_xy= mask_correlation(tract_mask, comp, tract)
            correlation_dict[name].append(stats.pearsonr(corr_xy['tract'][0], 
                                                         corr_xy['component'][0])[0])
    print(f'Saving csv to {options["csv"]}')
    csv_name = 'correlations_to_tracts.csv'
    if options['threshold']:
        csv_name = 'correlations_to_tracts_thresholded.csv'
    if options['csv_name']:
        csv_name = options['csv_name'] 
    df = pd.DataFrame(correlation_dict, index=[f'component_{comp}'for comp in range(0,len(correlation_dict['cbt_l']))]).T
    df.to_csv(f'{options["csv"]}/{csv_name}') 
    print('Done')

    
