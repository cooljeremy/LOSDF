# LOSDF: A Logical Optimization and Semantic Decoupling Framework for Question Answering in Multi-party Conversations

## Introduction

This project proposes a Logical Optimization and Semantic Decoupling Framework for Multi-party Dialogue Q&A (LOSDF).LOSDF enhances the model's ability to identify logical structures and transitions in a dialogue by integrating multi-party attentional mechanisms to manage dynamic and complex information flows across multiple speakers. In addition, the challenge of different linguistic styles between speakers is addressed by incorporating semantic rewriting mechanisms to ensure that responses are not only contextually relevant but also stylistically consistent with the speaker's expression. The information decoupling module is a key innovation of LOSDF, which excels at separating relevant information from the pervasive noise common in MPCs, thus extracting information relevant to the current query more accurately and efficiently.

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
│   ├── dream/
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

## Data preparation

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
1. Modify configuration: Modify the configuration parameters in config.py file, such as dataset path, model parameters, training parameters, etc. as needed.  
2. Run the training script.
```
python scripts/train.py
```

## Model evaluation
Run the evaluation script:
```
python scripts/evaluate.py
```
The evaluation results will be output to the console and a log will be generated in the logs/ folder.
