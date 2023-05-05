import numpy as np 
from utils.utils import create_directory

def get_len_and_step_size(attributions, slices=5):
    attr_len  = len(attributions)
    step_size = int(attr_len / slices)
    return attr_len,step_size


def create_pointwise_explanations(attributions):
    output = []
    for split in attributions:
        explanations = []
        for ts in split: 
            x_values = ts[1]
            attributions = ts[2]
            explanations.append(np.concatenate((attributions,np.array([x_values])), axis=None))    
        output.append(np.array(explanations))
    return output

def save_explanations(data, root_dir, archive_name, appendix, dataset_name):
    train_explanation,test_explanation = data
    print(train_explanation.shape, test_explanation.shape)
    dir_path = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/' + appendix + '/' 
    create_directory(dir_path)
    np.savetxt(dir_path + dataset_name + "_TRAIN", train_explanation, delimiter=',')
    np.savetxt(dir_path + dataset_name + "_TEST", test_explanation, delimiter=',')
    print("Successfully created explanation done.")