
from ..utils.utils import create_directory
from ..utils.utils import read_dataset
from ..utils.utils import calculate_pointwise_attributions
from ..utils.explanations import create_pointwise_explanations, save_explanations
from ..main_mt import fit_classifier


def experiment_1(datasets: list,
                 gammas : list, 
                 root_dir: str,
                 archive_name: str,
                 clsasifiers: None): 
    
    CLASSIFIERS_STL = ['fcn']

 
    for dataset_name in datasets: 

        data_dest = 'minmax'

        for classifier in classifiers: 
        
        # Run Single Task Learner with explanations
        datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original')

        
        fit_classifier()
        att = calculate_pointwise_attributions(root_dir, archive_name, classifier, dataset_name, data_source, mode, task=1)
        exp = create_pointwise_explanations(att, minmax_norm=True)
        save_explanations(exp, root_dir, archive_name, data_dest, dataset_name)

