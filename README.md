# Report

## Introduction

This is a solution to the problem of detecting misinformation claims. The task is a classification problem: assigning a label to each sentence. The proposed solution is based on the publicly available causal LLM Qwen/Qwen2.5-3B.

In this approach, a prompt is constructed using possible classes, examples (used in the second and third stages), and the input text to be classified. The prompt is then passed to the causal model, which generates an output from which the label is extracted.

## Stages of the Experiment

This experiment consists of three stages:

1. **Zero-Shot Prompting**  
   In this stage, the prompt is formed using the list of labels and the input text (e.g., `"0_0: NoClaim, 1_1: ..."`) only. I noticed a latent order in the codes, so I arranged them in the prompt by sorted code values.

2. **Few-Shot Prompting**  
   A few examples (texts with claim code and claim) are added to the prompt, specifically, one example per class from the training set. Since there are 18 classes, this becomes an 18-shot prompt. This longer prompt increases generation time but provides the model with representative examples for all classes.

3. **Few-Shot Prompting with RAG (Retrieval-Augmented Generation)**  
   This stage uses a RAG-based approach to retrieve the top-k (here, top-5) most relevant examples from the training set. Instead of random examples, the 5 closest samples to the query (measured via cosine similarity in embedding space) are included in the prompt. Embeddings are generated using the lightweight and efficient `all-MiniLM-L6-v2` model from `sentence-transformers`.

## Results

To compare zero-shot, few-shot, and RAG-based prompting, I used `sklearn`'s classification report, which reports precision, recall, F1-score, and accuracy.  

- **Zero-Shot**:  
  The model shows poor overall performance, though precision and F1-score are relatively high for the "no-claim" class. This suggests the model lacks strong task-specific knowledge in its pretraining data and defaults to predicting "no-claim".

- **Few-Shot**:  
  When one example per class is included, the model performs better across multiple classes. The macro F1-score improves, but the weighted F1-score decreases (alongside accuracy) due to class imbalance, most samples belong to the "no-claim" class. Including examples helps the model identify minority classes at the expense of the majority class (as reflected in severe reduced recall for "no-claim").

- **Few-Shot with RAG**:  
  This setup yields the best performance, improving both macro and weighted F1-scores. We also see model performs relatively good even in minority classes.  RAG effectively retrieves relevant examples that help guide the model in producing better predictions.

## Conclusion 
The using few-shot prompting with RAG could successfully improve the performance of classifying claims. Some ideas for extending this project are: fine-tuning the model with LoRa, adopting encoders for this problem. It is also intersting to investigate analysis of wrong-classified samples, and see how model confuses classes, and if the performance can be improved.

## Dataset

The dataset contains `"text"`, `"sub-claim"`, and `"sub-claim-code"`. The dataset is highly imbalanced toward 'no-claim' samples. The model is trained to assign the correct label (`sub-claim-code`) based on the input `"text"`. The dataset is available at: 
 https://huggingface.co/datasets/murathankurfali/ClimateEval/tree/main/exeter/sub_claim

## Model

To use a different model, modify the `model_id` parameter to another available CausalLM model from HuggingFace. 

## Assumptions and TODOs

- The model is assumed to generate a single label, chosen from the labels listed in the prompt.
- I used RAG. instead of finetuning with methods like LoRa and QLoRa, to use the power of pretrained causal models to its maximum extent. I usually don't use causal models for classification tasks, as they are inherently generative. Hence I preferred the idea of using RAG, instead of finetunning (e.g., LoRa) a causal LLM for classification task. However, I know these models are very effective for classification problems, and also this solution demonstrates they can be effective for this problem. For high-performance classification with limited training data, I recommend fine-tuning encoder-based models (e.g., BERT with a classification head).
- Other setups of improving performance with LoRa and QLoRa can be tested.
- This solution is a proof of concept, and the choice of models has to be investigated further.
- The code is a proof of concept can be improved and be optimized. Some examples are, do inference with  batched samples, pre-compute the RAG encodings and prompts of test set. 



## Running the Code

The code is written in Google Colab, and it is suggested to run it on Google Colab. To run it locally (preferably with a GPU), install the following dependencies (preferably in a virtual environment):

```bash
pip install 'tensorflow[and-cuda]'  # or install PyTorch from https://pytorch.org/get-started/locally/
pip install transformers
pip install sentence-transformers
pip install scikit-learn
```


## Results

### Results of Zero-Shot

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0_0   | 0.90      | 0.54   | 0.68     | 1753    |
| 1_1   | 0.07      | 0.96   | 0.13     | 51      |
| 1_2   | 0.50      | 0.14   | 0.22     | 21      |
| 1_3   | 0.14      | 0.60   | 0.23     | 30      |
| 1_4   | 0.15      | 0.44   | 0.23     | 68      |
| 1_6   | 0.33      | 0.04   | 0.07     | 26      |
| 1_7   | 0.00      | 0.00   | 0.00     | 64      |
| 2_1   | 0.54      | 0.10   | 0.18     | 124     |
| 2_3   | 0.00      | 0.00   | 0.00     | 48      |
| 3_1   | 0.41      | 0.27   | 0.33     | 26      |
| 3_2   | 0.00      | 0.00   | 0.00     | 49      |
| 3_3   | 0.00      | 0.00   | 0.00     | 46      |
| 4_1   | 0.15      | 0.78   | 0.25     | 64      |
| 4_2   | 0.20      | 0.06   | 0.09     | 34      |
| 4_4   | 0.00      | 0.00   | 0.00     | 39      |
| 4_5   | 0.00      | 0.00   | 0.00     | 36      |
| 5_1   | 0.22      | 0.42   | 0.29     | 224     |
| 5_2   | 0.00      | 0.00   | 0.00     | 195     |

**Accuracy:** 0.42  
**Macro avg:** Precision: 0.20, Recall: 0.24, F1: 0.15  
**Weighted avg:** Precision: 0.61, Recall: 0.42, F1: 0.46

### Results of Few Random Shots

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0_0   | 0.93      | 0.19   | 0.32     | 1753    |
| 1_1   | 0.11      | 0.80   | 0.19     | 51      |
| 1_2   | 0.08      | 0.86   | 0.15     | 21      |
| 1_3   | 0.64      | 0.23   | 0.34     | 30      |
| 1_4   | 1.00      | 0.01   | 0.03     | 68      |
| 1_6   | 0.61      | 0.42   | 0.50     | 26      |
| 1_7   | 0.33      | 0.08   | 0.13     | 64      |
| 2_1   | 0.67      | 0.11   | 0.19     | 124     |
| 2_3   | 0.06      | 0.06   | 0.06     | 48      |
| 3_1   | 0.50      | 0.27   | 0.35     | 26      |
| 3_2   | 0.33      | 0.29   | 0.30     | 49      |
| 3_3   | 0.56      | 0.20   | 0.29     | 46      |
| 4_1   | 0.27      | 0.12   | 0.17     | 64      |
| 4_2   | 0.09      | 0.71   | 0.15     | 34      |
| 4_4   | 0.21      | 0.08   | 0.11     | 39      |
| 4_5   | 0.21      | 0.56   | 0.31     | 36      |
| 5_1   | 0.24      | 0.63   | 0.34     | 224     |
| 5_2   | 0.18      | 0.66   | 0.28     | 195     |

**Accuracy:** 0.27  
**Macro avg:** Precision: 0.39, Recall: 0.35, F1: 0.23  
**Weighted avg:** Precision: 0.70, Recall: 0.27, F1: 0.29


### Results of Few Shots with RAG

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0_0   | 0.93      | 0.78   | 0.85     | 1753    |
| 1_1   | 0.31      | 0.76   | 0.45     | 51      |
| 1_2   | 0.35      | 0.71   | 0.47     | 21      |
| 1_3   | 0.59      | 0.87   | 0.70     | 30      |
| 1_4   | 0.53      | 0.57   | 0.55     | 68      |
| 1_6   | 0.61      | 0.96   | 0.75     | 26      |
| 1_7   | 0.79      | 0.77   | 0.78     | 64      |
| 2_1   | 0.71      | 0.55   | 0.62     | 124     |
| 2_3   | 0.55      | 0.58   | 0.57     | 48      |
| 3_1   | 0.50      | 0.54   | 0.52     | 26      |
| 3_2   | 0.78      | 0.96   | 0.86     | 49      |
| 3_3   | 0.85      | 0.96   | 0.90     | 46      |
| 4_1   | 0.29      | 0.55   | 0.38     | 64      |
| 4_2   | 0.30      | 0.50   | 0.38     | 34      |
| 4_4   | 0.55      | 0.56   | 0.56     | 39      |
| 4_5   | 0.52      | 0.67   | 0.59     | 36      |
| 5_1   | 0.55      | 0.73   | 0.63     | 224     |
| 5_2   | 0.56      | 0.55   | 0.56     | 195     |

**Accuracy:** 0.74  
**Macro avg:** Precision: 0.57, Recall: 0.70, F1: 0.62  
**Weighted avg:** Precision: 0.79, Recall: 0.74, F1: 0.75





# License
All rights reserved.


