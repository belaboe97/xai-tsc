import numpy as np 
from utils.utils import create_directory

def get_len_and_step_size(attributions, slices=5):
    attr_len  = len(attributions)
    step_size = int(attr_len / slices)
    return attr_len,step_size


def highest_mean_attribution(att,slices): 
    attr_len, step_size = get_len_and_step_size(att,slices)
    candidates = []
    for x in range(0,attr_len,step_size):
        candidates.append(att[x:x+step_size-1].mean())
    return np.argmax(candidates)

def create_explanations(attributions,slices):
    output = []
    for split in attributions:#
        explanations = []
        for ts in split: 
            x_values = ts[6]
            label = highest_mean_attribution(ts[3],slices)
            explanations.append(np.concatenate((label,x_values), axis=None))    
        output.append(np.array(explanations))
    return output

def create_pointwise_explanations(attributions):
    output = []
    for split in attributions:
        explanations = []
        for ts in split: 
            y_values = ts[0]
            attributions = ts[2]
            explanations.append(np.concatenate((y_values,attributions), axis=None))    
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