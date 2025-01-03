# LOSDF: A Logical Optimization and Semantic Decoupling Framework for Question Answering in Multi-party Conversations

## Introduction

This paper proposes a Logical Optimization and Semantic Decoupling Framework for Multi-party Dialogue Q&A (LOSDF).LOSDF enhances the model's ability to identify logical structures and transitions in a dialogue by integrating multi-party attentional mechanisms to manage dynamic and complex information flows across multiple speakers. In addition, the challenge of different linguistic styles between speakers is addressed by incorporating semantic rewriting mechanisms to ensure that responses are not only contextually relevant but also stylistically consistent with the speaker's expression. The information decoupling module is a key innovation of LOSDF, which excels at separating relevant information from the pervasive noise common in MPCs, thus extracting information relevant to the current query more accurately and efficiently.

## Code Structure
```
LOSDF/
├── data/                     # data set
│   ├── molweni/
│   │   ├── train.json
│   │   ├── dev.json
│   │   └── test.json
│   ├── friendsqa/
│   │   ├── train.json
│   │   ├── dev.json
│   │   └── test.json
│   └── dailydialog/
│       ├── train.json
│       ├── dev.json
│       └── test.json
├── scripts/                  
│   ├── preprocess_data.py    
│   ├── train.py              
│   ├── evaluate.py           
│   └── inference.py          
├── models/                   # model code
│   ├── __init__.py
│   ├── losdf.py              # LOSDF Model
│   ├── components.py         # model component (Rewriting, Multi-Party Attention, Decoupling)
│   └── utils.py              # instrumented function
├── requirements.txt          # dependency package
├── config.py                 # configuration file
├── README.md                 # documentation
└── .gitignore                
```


## Environment Configuration

### 1.Creating a Virtual Environment：
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

### 2.Installation of dependency packages:
pip install -r requirements.txt

## Data Preparation

The following three datasets are used in this project.  
Molweni: a multi-party conversational Q&A dataset derived from the Ubuntu forums, which is skewed towards technical discussions.  
FriendsQA: a multi-party Q&A dataset derived from the TV series “Friends”, favoring everyday conversations.  
DailyDialog: A high-quality multi-round dialog dataset covering a variety of everyday topics.  

Please download the dataset to the data/ folder and organize it according to the following structure：
```
data/
├── molweni/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── friendsqa/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── dream/
│   ├── train.json
│   ├── dev.json
│   └── test.json
└── dailydialog/
    ├── train.json
    ├── dev.json
    └── test.json
```
Data is in a uniform JSON format:
```
{
    "dialog_id": "12345",
    "context": [
        {"speaker": "A", "utterance": "Hello, how are you?"},
        {"speaker": "B", "utterance": "I'm fine, thanks. And you?"},
        {"speaker": "A", "utterance": "I'm good too."}
    ],
    "question": "How is speaker B doing?",
    "answer": "I'm fine, thanks."
}
```
Usage:  
1.Download the raw dataset to the data/ folder and organize it according to the directory structure described in README.md.  
2.Modify the configuration parameters in the config.py file as needed.  
3.Run the following command for data preprocessing:  
```
python scripts/preprocess_data.py --dataset molweni
```
Specify the raw data and output path (optional):
```
python scripts/preprocess_data.py --dataset all --data_dir /path/to/raw/data --output_dir /path/to/processed/data
```
The preprocessed data will be saved in the corresponding subfolders under the data/ folder, e.g. data/molweni/train.json, data/molweni/dev.json, data/molweni/test.json



## Model Training
1. Make sure you have prepared the data and code as per the previous steps.  
2. Modify the configuration parameters in the config.py file as needed.
3. Run the following command to start training:  
Using losdf models trained on the molweni dataset  
```
python scripts/train.py --dataset_name molweni --model_type losdf
```
Training on the friendsqa dataset using the electra_baseline model  
```
python scripts/train.py --dataset_name friendsqa --model_type electra_baseline
```
Using losdf models trained on the dailydialog dataset 
```
python scripts/train.py --dataset_name dailydialog --model_type losdf
```


## Model Evaluation
1. Make sure you have trained the LOSDF model and got the saved model parameters.  
2. Run the following command for evaluation:  
```
python scripts/evaluate.py --model_path /path/to/your/model --dataset_name molweni --split test
```
Replace /path/to/your/model with the path where your actual model is stored, molweni with the dataset you want to evaluate, and test with dev or test.  

The evaluation results will be output to the console and a log will be generated in the logs/ folder.

Example:  
```
python scripts/evaluate.py --model_path saved_models/best_model --dataset_name molweni --split test --output_file predictions.json
```
The model will be loaded in the saved_models/best_model path, evaluated on the test set of the molweni dataset, and the predictions will be saved in predictions.json.


## Inference
### 1. Non-interactive Mode:
```
python scripts/inference.py --model_path /path/to/your/model --context_file data/test_context.json --question "What is the answer?
```
Replace /path/to/your/model with your actual model save path. data/test_context.json is a JSON file containing the context of the conversation in the following format:  
```
[
    {"speaker": "A", "utterance": "Hello, how are you?"},
    {"speaker": "B", "utterance": "I'm fine, thanks. And you?"},
    {"speaker": "A", "utterance": "I'm good too."}
]
```

### 2. Interaction Model:
```
python scripts/inference.py --model_path /path/to/your/model --interactive
```

### Example:  
Non-interactive mode example:  
The data/test_context.json file has the following contents:  
```
[
    {"speaker": "A", "utterance": "What's your favorite color?"},
    {"speaker": "B", "utterance": "My favorite color is blue."}
]
```
Run the following command:  
```
python scripts/inference.py --model_path saved_models/best_model --context_file data/test_context.json --question "What is B's favorite color?"
```
Output:
```
Context: [{'speaker': 'A', 'utterance': "What's your favorite color?"}, {'speaker': 'B', 'utterance': 'My favorite color is blue.'}]
Question: What is B's favorite color?
Answer: blue.
```

Examples of interaction patterns:  
```
python scripts/inference.py --model_path saved_models/best_model --interactive
```
Then follow the prompts to enter:  
```
Entering interactive mode. Type 'exit' to quit.
Enter the dialog context (or 'load' to load from file): load
Enter the context file path: data/test_context.json
Enter your question: What is B's favorite color?
Answer: blue.
Enter the dialog context (or 'load' to load from file):
```
