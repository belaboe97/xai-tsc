ITERATIONS = 5  # nb of random runs for random initializations

ARCHIVE_NAMES = ['UCRArchive_2018']

CLASSIFIERS = ['fcn','resnet']#'fcn']  #'mlp', 'resnet', 'tlenet', 'mcnn', 'twiesn', 'encoder', 'mcdcnn', 'cnn', 'inception']


CAM_LAYERS = {
    'fcn' : {
        'gap_layer' :  'activation_2',
        'task_1' :  'task_1_output',
        'task_2' :  'task_2_output',
    },
    'resnet' : {
        'gap_layer' :  'activation_8',
        'task_1' :  'task_1_output',
        'task_2' :  'task_2_output', 
    }
}
