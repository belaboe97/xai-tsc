import numpy as np 
from utils.utils import create_directory



def create_pointwise_explanations(attributions, minmax_norm = False):
    output = []
    for split in attributions:
        explanations = []
        for ts in split: 
            x_values = ts[1]
            if minmax_norm:
                attributions = (ts[2] - np.min(ts[2])) / (np.max(ts[2]) - np.min(ts[2]))
            else: 
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