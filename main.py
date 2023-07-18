from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
import os
import numpy as np
import sys
import sklearn
from utils.constants import CLASSIFIERS
from utils.constants import ITERATIONS
from utils.explanations import calculate_cam_attributions
from utils.explanations import calculate_ig_attributions
from utils.explanations import create_explanations
from utils.explanations import save_explanations
import tensorflow as tf
from utils.classifiers import fit_classifier

###### SETTINGS


if os.getenv("COLAB_RELEASE_TAG"):
    print("Google Colab Environment detected")
    root_dir =  "/content/drive/My Drive/master thesis/code/xai-tsc"
    EPOCHS = 400
    BATCH_SIZE = 16
    print('Epochs',EPOCHS, 'Batch size', BATCH_SIZE)
else: 
    print("Local Environment detected")
    root_dir = "G:/Meine Ablage/master thesis/code/xai-tsc"
    EPOCHS = 3
    BATCH_SIZE = 16
    print('Epochs',EPOCHS, 'Batch size', BATCH_SIZE)

#Set random seed 
#https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed


SEED = 0
DATASET_NAMES = ['GunPoint', 'ECG200']#, 'Beef', 'GunPoint']#,'ECG200']#'Beef','Coffee' ,'GunPoint']
LOSSES = ['mse']#, 'cosinesim']
DATASCALING = 'raw' #minmax
ITERATIONS = 6

print(f'In fixed SEED mode: {SEED}')
print(f'Epochs for each classifier is set to {EPOCHS} and Batchsize set to {BATCH_SIZE}')

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def set_global_determinism(seed=SEED):
    """ set_seeds(seed=seed)

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
        #tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    """
    print("set global determinism")


#set_global_determinism(SEED)

#ARGS  
mode = sys.argv[1]
if mode == 'singletask' or mode == 'multitask':
    archive_name = sys.argv[2]
    dataset_name = sys.argv[3]
    classifier_name = sys.argv[4]
    itr = sys.argv[5] if sys.argv[5] != '_itr_0' else ''
    data_source = sys.argv[6] 
    data_dest = sys.argv[7]
    gamma = 0.5 if sys.argv[8] == None else sys.argv[8]
    gamma = np.float64(gamma)
    classifier = classifier_name + '_' + str(gamma) 


def output_path():
    output_directory = f'{root_dir}/results/{archive_name}/{dataset_name}/{classifier_name.split("_")[0]}/{classifier}/{data_source}/' 
    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name, itr)
    return output_directory, test_dir_df_metrics


if mode == 'singletask':

    output_directory, test_dir_df_metrics = output_path()

    if False:#os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:
        create_directory(output_directory)

    print(output_directory)

    datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original', 1)[dataset_name]
    fit_classifier(classifier_name, mode, datasets_dict, None, 
                   output_directory, 'mse', gamma, EPOCHS, BATCH_SIZE)
    
    #Integrated Gradients
    att = calculate_ig_attributions(root_dir, archive_name, classifier, dataset_name, data_source)
    exp = create_explanations(att, minmax_norm=False)
    save_explanations(exp, root_dir, archive_name, 'fcn_ig_raw', dataset_name)
    
    #Class Activation Mapping
    att = calculate_cam_attributions(root_dir, archive_name, classifier, dataset_name, data_source)
    exp = create_explanations(att, minmax_norm=False)
    save_explanations(exp, root_dir, archive_name, 'fcn_cam_raw', dataset_name)

if mode == 'multitask': 

    output_directory, test_dir_df_metrics = output_path()

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:
        create_directory(output_directory)

        datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original', 1)[dataset_name]
        x,_,_,_ =  datasets_dict
        datasets_dict_2 = read_dataset(root_dir, archive_name, dataset_name, data_source, len(x[0]))[dataset_name]
        fit_classifier(classifier_name, mode, datasets_dict, datasets_dict_2, 
                    output_directory,  'mse', gamma, EPOCHS, BATCH_SIZE) #mse #tf.keras.losses.CosineSimilarity(axis=1)
        

#Singletask
if mode == 'experiment_1': 

    archive_name  = 'ucr'

    for dataset_name in DATASET_NAMES: 
        datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original', 1)[dataset_name]


        for classifier_name in CLASSIFIERS:
            
            print("Classifier",classifier_name)
        
            best_acc = 0 
            best_model = 0
            
            for itr in range(ITERATIONS): 

                output_directory = f'{root_dir}/results/{archive_name}/{dataset_name}/' \
                    f'/experiment_1/{classifier_name}/{classifier_name}_{itr}/original/'  
                
                test_dir_df_metrics = output_directory + 'task1_df_metrics.csv'

                if os.path.exists(test_dir_df_metrics):
                    print('Already done')
                else:
                    create_directory(output_directory)
                
                    create_directory(output_directory)

                    acc = fit_classifier(classifier_name, 'singletask', datasets_dict, None, 
                                output_directory, None, 1, EPOCHS, BATCH_SIZE)
                    print(acc)
                    if best_acc < acc: best_acc = acc; best_model = itr

            best_classifier = f'{classifier_name}_{best_model}'


            att = calculate_cam_attributions(root_dir, archive_name, best_classifier, 
                                            dataset_name, 'original')
            
            exp = create_explanations(att, minmax_norm=False)
            save_explanations(exp, root_dir, archive_name, f'{classifier_name}_cam_raw', dataset_name)

            #Create Integrated Gradients Explanations
            #testing purpse
            best_classifier = f'{classifier_name}_{1}'
            att = calculate_ig_attributions(root_dir, archive_name, best_classifier, 
                                            dataset_name, 'original', task=0)
            exp = create_explanations(att, minmax_norm=False)
            save_explanations(exp, root_dir, archive_name, f'{classifier_name}_ig_raw', dataset_name)
        
            
        
            mtc_path  = f'{root_dir}/classifiers_mtl/{classifier_name}'


            for expl_type in ['fcn_ig_raw']:#,'resnet_ig_raw',]:#,'fcn_cam_raw']:,
                
                #assert same length for all ts
                exp_len = len(datasets_dict[0][0])
                datasets_dict_2 = read_dataset(root_dir, archive_name, dataset_name, expl_type, exp_len)[dataset_name]                    
                print(os.listdir(mtc_path))
                for mtclassifier in os.listdir(mtc_path):

                        if 'fcn_mt_conv' not in mtclassifier: 
                            continue

                        #check only for same explanation types
                        #avoid pycache
                        if not mtclassifier.startswith('_'):
                            for itr in range(ITERATIONS): 
                                
                                mt_classifier = mtclassifier.split('.')[0]
                                if mt_classifier.split('_')[0] not in expl_type: 
                                    print(mt_classifier.split('_')[0], expl_type, mt_classifier.split('_')[0] not in expl_type )
                                    continue

                                #if 'ae' not in mt_classifier: continue 
                                print(mt_classifier)
                                #Explanation Type 
                                output_directory = f'{root_dir}/results/{archive_name}/{dataset_name}/' \
                                f'/experiment_1/{classifier_name}/{mt_classifier}_{itr}/{expl_type}/'  

                                test_dir_df_metrics = output_directory + 'task1_df_metrics.csv'

                                print(test_dir_df_metrics)
                                if os.path.exists(test_dir_df_metrics):
                                    print('Already done')
                                else:
                                    create_directory(output_directory)
                                    print(mt_classifier, 'multitask', datasets_dict, datasets_dict_2, 
                                                output_directory, 
                                                'mse', 1, EPOCHS, BATCH_SIZE)
                                    fit_classifier(mt_classifier, 'multitask', datasets_dict, datasets_dict_2, 
                                                output_directory, 
                                                'mse', 1, EPOCHS, BATCH_SIZE)
                            

            

if mode == 'experiment_2': 

    archive_name = 'ucr'
    GAMMAS = [ 0.5, 0.25]#, 0.5, 0.25]

    for dataset_name in DATASET_NAMES: 

        datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original', 1)[dataset_name]

        for expl_type in ['resnet_ig_trf']:#,'resnet_ig_raw']:#,'fcn_cam_raw']:,

            #assert same length for all ts
            exp_len = len(datasets_dict[0][0])
            datasets_dict_2 = read_dataset(root_dir, archive_name, dataset_name, expl_type, exp_len)[dataset_name]                    
            for classifier_name in CLASSIFIERS: 
                
                mtc_path  = f'{root_dir}/classifiers_mtl/{classifier_name}'

                #assert same length for all ts
                exp_len = len(datasets_dict[0][0])
                datasets_dict_2 = read_dataset(root_dir, archive_name, dataset_name, expl_type, exp_len)[dataset_name]                    
                print(os.listdir(mtc_path))
                for mtclassifier in os.listdir(mtc_path):
                        
                        if 'resnet_mt_ae' not in mtclassifier: continue
                        #check only for same explanation types
                        #avoid pycache
                        if not mtclassifier.startswith('_'): 
                            for gamma in GAMMAS:  
                                for itr in range(ITERATIONS): 
                                    
                                    mt_classifier = mtclassifier.split('.')[0]
                                    if mt_classifier.split('_')[0] not in expl_type: 
                                        print(mt_classifier.split('_')[0], expl_type, mt_classifier.split('_')[0] not in expl_type )
                                        continue

                                    #if 'ae' not in mt_classifier: continue 
                                    print(mt_classifier)
                                    #Explanation Type 

                                    output_directory = f'{root_dir}/results/{archive_name}/{dataset_name}/' \
                                                        f'/experiment_2/{classifier_name}/{mt_classifier}_{gamma}_{itr}/{expl_type}/'   

                                    test_dir_df_metrics = output_directory + 'task1_df_metrics.csv'

                                    print(test_dir_df_metrics)
                                    if os.path.exists(test_dir_df_metrics):
                                        print('Already done')
                                    else:
                                        create_directory(output_directory)

                                        fit_classifier(mt_classifier, 'multitask', datasets_dict, datasets_dict_2, output_directory, 
                                                    'mse', gamma, EPOCHS, BATCH_SIZE)




if mode == 'experiment_3': 

    archive_name = 'ucr'
    GAMMAS = [0.75]

    for dataset_name in DATASET_NAMES: 

        datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original', 1)[dataset_name]

        for expl_type in ['fcn_ig_raw']:#,'resnet_ig_raw']:#,'resnet_ig_raw']:#,'fcn_cam_raw']:,

            #assert same length for all ts
            exp_len = len(datasets_dict[0][0])
            datasets_dict_2 = read_dataset(root_dir, archive_name, dataset_name, expl_type, exp_len)[dataset_name]                    
            for classifier_name in CLASSIFIERS: 
                
                mtc_path  = f'{root_dir}/classifiers_mtl_cascade/{classifier_name}'

                #assert same length for all ts
                exp_len = len(datasets_dict[0][0])
                datasets_dict_2 = read_dataset(root_dir, archive_name, dataset_name, expl_type, exp_len)[dataset_name]                    
                print(os.listdir(mtc_path))
                for mtclassifier in os.listdir(mtc_path):
                        #check only for same explanation types
                        #avoid pycache
                        if not mtclassifier.startswith('_'): 
                            for gamma in GAMMAS:  
                                for itr in range(ITERATIONS): 
                                    
                                    mt_classifier = mtclassifier.split('.')[0]
                                    if mt_classifier.split('_')[0] not in expl_type: 
                                        print(mt_classifier.split('_')[0], expl_type, mt_classifier.split('_')[0] not in expl_type )
                                        continue

                                    #if 'ae' not in mt_classifier: continue 
                                    print(mt_classifier)
                                    #Explanation Type 

                                    output_directory = f'{root_dir}/results/{archive_name}/{dataset_name}/' \
                                                        f'/experiment_3/{classifier_name}/{mt_classifier}_{gamma}_{itr}/{expl_type}/'   

                                    test_dir_df_metrics = output_directory + 'task1_df_metrics.csv'

                                    print(test_dir_df_metrics)
                                    if os.path.exists(test_dir_df_metrics):
                                        print('Already done')
                                    else:
                                        create_directory(output_directory)

                                        fit_classifier(mt_classifier, 'multitask', datasets_dict, datasets_dict_2, output_directory, 
                                                    'mse', gamma, EPOCHS, BATCH_SIZE)



if mode == 'experiment_4': 

    archive_name = 'ucr'
    GAMMAS = [0.75]


    for dataset_name in DATASET_NAMES: 

        datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original', 1)[dataset_name]

        for expl_type in ['fcn_ig_raw']:#,'resnet_ig_raw']:#,'resnet_ig_raw']:#,'fcn_cam_raw']:

            #assert same length for all ts
            exp_len = len(datasets_dict[0][0])
            datasets_dict_2 = read_dataset(root_dir, archive_name, dataset_name, expl_type, exp_len)[dataset_name]                    
            for classifier_name in CLASSIFIERS: 

                
                mtc_path  = f'{root_dir}/classifiers_mtl_iterative/{classifier_name}'
    
                #assert same length for all ts
                exp_len = len(datasets_dict[0][0])
                datasets_dict_2 = read_dataset(root_dir, archive_name, dataset_name, expl_type, exp_len)[dataset_name]                    
                print(os.listdir(mtc_path))
                for mtclassifier in os.listdir(mtc_path):
                        
                        if 'freeze' not in mtclassifier: 
                            continue
                        #check only for same explanation types
                        #avoid pycache
                        if not mtclassifier.startswith('_'): 
                            for gamma in GAMMAS:  
                                for itr in range(ITERATIONS): 
                                    
                                    mt_classifier = mtclassifier.split('.')[0]
                                    if mt_classifier.split('_')[0] not in expl_type: 
                                        print(mt_classifier.split('_')[0], expl_type, mt_classifier.split('_')[0] not in expl_type )
                                        continue

                                    #if 'ae' not in mt_classifier: continue 
                                    print(mt_classifier)
                                    #Explanation Type 

                                    output_directory = f'{root_dir}/results/{archive_name}/{dataset_name}/' \
                                                        f'/experiment_4/{classifier_name}/{mt_classifier}_{gamma}_{itr}/{expl_type}/'   

                                    test_dir_df_metrics = output_directory + 'task1_df_metrics.csv'

                                    print(test_dir_df_metrics)
                                    if os.path.exists(test_dir_df_metrics):
                                        print('Already done')
                                    else:
                                        create_directory(output_directory)

                                        fit_classifier(mt_classifier, 'multitask', datasets_dict, datasets_dict_2, output_directory, 
                                                    'mse', gamma, EPOCHS, BATCH_SIZE)
