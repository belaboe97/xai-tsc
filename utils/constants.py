ITERATIONS = 5  # nb of random runs for random initializations

ARCHIVE_NAMES = ['UCRArchive_2018']

CLASSIFIERS = ['fcn','resnet']#'fcn']  #'mlp', 'resnet', 'tlenet', 'mcnn', 'twiesn', 'encoder', 'mcdcnn', 'cnn', 'inception']


CAM_LAYERS = {
    'fcn' : {
        'last_conv_layer' :  'shared_l9',
        'task_1' :  'task_1_output',
        'task_2' :  'task_2_output',
    },
    'resnet' : {
        'last_conv_layer' :  'shared_l33',
        'task_1' :  'task_1_output',
        'task_2' :  'task_2_output', 
    }
}
