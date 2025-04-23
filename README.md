# DistilBERT Fine-Tuning for News Classification

This project implements fine-tuning of the DistilBERT model on the AG News dataset for text classification. The system categorizes news articles into four classes: World, Sports, Business, and Sci/Tech.

## Project Overview

The project demonstrates the complete workflow for fine-tuning a pre-trained language model:

1. **Environment Setup**: Configuration of dependencies and libraries
2. **Dataset Preparation**: Loading and preprocessing the AG News dataset
3. **Model Selection**: Using DistilBERT as an efficient BERT variant
4. **Fine-Tuning Setup**: Training configuration with hyperparameter optimization
5. **Model Evaluation**: Comprehensive performance assessment
6. **Error Analysis**: Identification of misclassification patterns
7. **Inference Pipeline**: Ready-to-use classification system

## Dataset

The **AG News** dataset contains news articles from four categories:
- World (0)
- Sports (1)
- Business (2)
- Sci/Tech (3)

The dataset includes both titles and descriptions of news articles, with 120,000 training examples and 7,600 test examples.

## Model Selection

**DistilBERT** was selected for this project for several key reasons:

- **Efficiency**: 40% smaller than BERT-base while maintaining 97% of its performance
- **Speed**: 60% faster training and inference than BERT
- **Compatibility**: Fully compatible with the Hugging Face ecosystem
- **Resource Requirements**: Lower memory footprint, ideal for deployment
- **Task Appropriateness**: Well-suited for text classification with minimal modifications

## Fine-Tuning Setup

The fine-tuning process used:

- **Tokenizer**: DistilBERT tokenizer with maximum sequence length of 128
- **Model Architecture**: DistilBERT with a classification head (4 classes)
- **Training Environment**: Google Colab with T4 GPU
- **Training Framework**: Hugging Face Transformers with PyTorch backend

## Hyperparameter Optimization

The hyperparameter search focused on finding the optimal learning rate:

| Learning Rate | Eval Loss |
|---------------|-----------|
| 5e-5          | 0.2695    |
| 1e-4          | 0.2935    |
| 2e-4          | 0.4029    |

Based on these results, **5e-5** was selected as the optimal learning rate, providing the best generalization on the validation set.

Other fixed hyperparameters:
- Batch size: 8
- Epochs: 0.2 (partial epoch for quick experimentation)
- Optimizer: AdamW
- Mixed precision training (FP16)

## Model Evaluation

Performance comparison:

| Model Version      | Accuracy |
|--------------------|---------|
| Baseline (Untrained) | 27.10%  |
| Fine-tuned          | 89.82%  |

This demonstrates an improvement of **62.72 percentage points** after fine-tuning, confirming the effectiveness of task-specific adaptation.

## Error Analysis

Analysis of misclassified examples revealed several patterns:

1. **Label Confusion**:
   - Confusion between `World` and `Business` categories for articles about government economic policy
   - Sports-related business news being classified as `Sports` instead of `Business`
   - Tech companies being classified as `Sci/Tech` rather than `Business`

2. **Contextual Challenges**:
   - Articles with domain overlap (e.g., "NBC exec survives crash" containing media figure)
   - Headlines with insufficient context to make fine-grained distinctions

3. **Entity and Keyword Bias**:
   - Over-association of entity names (e.g., "Yankees") with specific categories
   - Government/financial terms biasing toward `Business` classification

## Inference Pipeline

The system provides a simple inference function to classify new text:

```python
def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                      padding=True, max_length=128).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted = torch.argmax(outputs.logits, dim=-1).item()
    return predicted
```

## Future Improvements

Based on error analysis, potential improvements include:

- Using longer article summaries instead of just headlines
- Incorporating entity-aware embeddings
- Fine-tuning a larger model (e.g., BERT-base)
- Introducing label smoothing or class-weighted loss functions
- Data augmentation with examples that have overlapping topics

## Reproduction Instructions

1. **Environment Setup**:
   ```
   pip install transformers datasets peft accelerate bitsandbytes
   ```

2. **Running the Code**:
   - The notebook is designed to run in Google Colab with GPU support
   - All cells should be executed sequentially
   - Hyperparameter optimization can be extended by modifying the `learning_rates` list

3. **Dependencies**:
   - transformers
   - datasets
   - torch
   - pandas
   - scikit-learn

## License

This project is provided for educational purposes. The AG News dataset and pre-trained models are subject to their respective licenses.

## Acknowledgments

- Hugging Face for the Transformers library and datasets
- Google Colab for GPU resources
- The creators of the AG News dataset
