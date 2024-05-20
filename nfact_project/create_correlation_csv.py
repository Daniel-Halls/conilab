import pandas as pd
import nilearn.image as img
from nilearn.datasets import load_mni152_template
import scipy.stats as stats
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
                        help='Whether to threshold the image components before running correlations',
                        action='store_true')
    option.add_argument('--csv_name',
                        dest='csv_name',
                        help='What to name the csv. Default is correlations_to_tracts or correlations_to_tracts_thresholded',
                        type=str)
    return vars(option.parse_args())

if __name__=='__main__':
    options = args()
    nfm = img.load_img(options['image'])
    template = load_mni152_template(resolution=1)
    atlas = glob.glob(f"{options['atlas']}/*.nii.gz")
    pattern = re.compile(r'/(?P<word>[^/]+)\.nii\.gz')
    key = [match.group('word') for path in atlas if (match := pattern.search(path))]
    correlation_dict = dict(zip(key, [[] for at in key]))

    n_components = options['number_of_components']
    correlation_dict = dict(zip(key, [[] for at in key]))
    
    
    for tract_img in atlas:
        tract = img.load_img(tract_img)
        resample_tract = img.resample_to_img(tract, template, interpolation='nearest')
        name = re.findall(r'/(?P<word>[^/]+)\.nii\.gz', tract_img)[0]
        print(f'Working on {name}')
        for component in range(0, n_components):
            print(component, end='\r')
            comp = img.index_img(nfm, component)
            resample_component = img.resample_to_img(comp, template, interpolation='nearest')
            if options['threshold']:
                resample_component = img.threshold_img(img.resample_to_img(comp, template, 
                                                                           interpolation='nearest'), 
                                                        threshold=1)
            correlation_dict[name].append(stats.pearsonr(resample_component.get_fdata().ravel(), 
                                                         resample_tract.get_fdata().ravel())[0])
    print(f'Saving csv to {options["csv"]}')
    csv_name = 'correlations_to_tracts.csv'
    if options['threshold']:
        csv_name = 'correlations_to_tracts_thresholded.csv'
    if options['csv_name']:
        csv_name = options['csv_name'] 
    df = pd.DataFrame(correlation_dict, index=[f'component_{comp}'for comp in range(0,len(correlation_dict['cbt_l']))]).T
    df.to_csv(f'{options["csv"]}/{csv_name}') 
    print('Done')

    
