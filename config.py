class Config:
    def __init__(self):
        # Data set related
        self.data_dir = "data/"
        self.dataset_choices = ["molweni", "friendsqa", "dailydialog"]
        
        # Model Selection
        self.model_name = "losdf"  # choose: "losdf", "bert_baseline", "electra_baseline"
        
        # Pre-trained models
        self.bert_model = "bert-large-uncased" 
        self.electra_model = "google/electra-large-discriminator"
        
        # If another pre-training model is chosen, it can be modified.
        
        # LLM (For semantic rewriting)
        self.llm_model = "gpt-3.5-turbo"  
        self.llm_api_key = "YOUR_API_KEY" 

        # training parameter
        self.learning_rate = 1e-5
        self.batch_size = 32
        self.num_epochs = 10 # The number of training rounds can be dialed down appropriately
        self.logging_steps = 500 # Log prints are made every global step.
        
        # data parameter
        self.max_seq_length = 150  # Adjustment to the data set
        self.max_turns = 8  # Adjustment to the data set
        
        
        # Optimization methods for interaction vector computation
        self.max_interaction_range = 5  # K in the paper, only K speakers before and after are considered
        self.attention_threshold = 0.1  # Attention score threshold for pruning

        # Other parameters
        self.save_model = True # Whether to save the trained model
        self.save_path = "saved_models/" #Path where the model is saved
        self.load_model = False #Whether to load the trained model
        self.load_path = "" #Path for model loading
        self.do_lower_case = True
        self.gradient_accumulation_steps = 1 # Steps of gradient accumulation
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.warmup_steps = 0 #Number of steps for warm up
        self.seed = 42
        self.fp16 = False # Whether to use fp16 mixed precision
        self.fp16_opt_level = "O1" # Optimized level of mixing accuracy
        self.local_rank = -1 # Distributed training of rank
        self.n_gpu = 1 # Number of GPUs used
        self.device = "cuda" if self.n_gpu > 0 else "cpu"
        
        #ablation experiment
        self.ablation_methods = ["no_rewriting", "no_multi_party_attention", "no_information_decoupling"]