import os

class CFG:
    # Debugging
    debug = False
    
    # Plot training history
    training_plot = True
    
    # Weights and Biases logging
    wandb = True
    competition   = 'birdclef-2023' 
    _wandb_kernel = 'awsaf49'
    
    # Experiment name and comment
    exp_name = 'baseline-v2'
    comment = 'EfficientNetB0|FSR|t=10s|128x384|up_thr=50|cv_filter'
    
    # Notebook link
    notebook_link = 'https://www.kaggle.com/awsaf49/birdclef23-effnet-fsr-cutmixup-train/edit'
    
    # Verbosity level
    verbose = 0
    
    # Device and random seed
    device = 'TPU-VM'
    seed = 42
    
    # Input image size and batch size
    img_size = [128, 384]
    batch_size = 32
    upsample_thr = 50 # min sample of each class (upsample)
    cv_filter = True # always keeps low sample data in train
    
    # Inference batch size, test time augmentation, and drop remainder
    infer_bs = 2
    tta = 1
    drop_remainder = True
    
    # Number of epochs, model name, and number of folds
    epochs = 25
    model_name = 'EfficientNetB0'
    fsr = True # reduce stride of stem block
    num_fold = 5
    
    # Selected folds for training and evaluation
    selected_folds = [0]

    # Pretraining, neck features, and final activation function
    pretrain = 'imagenet'
    neck_features = 0
    final_act = 'softmax'
    
    # Learning rate, optimizer, and scheduler
    lr = 1e-3
    scheduler = 'cos'
    optimizer = 'Adam' # AdamW, Adam
    
    # Loss function and label smoothing
    loss = 'BCE' # BCE, CCE
    label_smoothing = 0.05 # label smoothing
    
    # Audio duration, sample rate, and length
    duration = 10 # second
    sample_rate = 32000
    target_rate = 8000
    audio_len = duration*sample_rate
    
    # STFT parameters
    nfft = 2048
    window = 2048
    hop_length = audio_len // (img_size[1] - 1)
    fmin = 20
    fmax = 16000
    normalize = True
    
    # Data augmentation parameters
    augment=True
    
    # Spec augment
    spec_augment_prob = 0.80
    
    mixup_prob = 0.65
    mixup_alpha = 0.5
    
    cutmix_prob = 0.0
    cutmix_alpha = 0.5
    
    mask_prob = 0.65
    freq_mask = 20
    time_mask = 30


    # Audio Augmentation Settings
    audio_augment_prob = 0.5
    
    timeshift_prob = 0.0
    
    gn_prob = 0.35

    # Data Preprocessing Settings
    base_path = '/kaggle/input/birdclef-2023'  # for server: base_path = '/data/zjh_data/program/ml_project_birdclef23/birdclef-2023'
    if not os.path.exists(base_path):
        base_path = '/data/zjh_data/program/ml_project_birdclef23/birdclef-2023'
    class_names = sorted(os.listdir('{}/train_audio'.format(base_path)))
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v:k for k,v in label2name.items()}

    # Training Settings
    target_col = ['target']
    tab_cols = ['filename']
    monitor = 'auc'
    
    ### add by plathzheng
    unilm_model_path = './pretrained_models/unilm/BEATs_iter3_plus_AS2M.pt'
    use_apex = True
    time_length = 10 # beats模型中，训练时，截取的音频片段时长
    ast_fix_layer = 3 # the parameters in layer<ast_fix_layer would be fixed, choosen from [0, 5], if ast_fix_layer>5 all param woudl be fixed