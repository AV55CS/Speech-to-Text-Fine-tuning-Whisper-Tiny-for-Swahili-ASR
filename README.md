# Swahili Speech Recognition - Whisper Fine-tuning

This repository contains code for fine-tuning OpenAI's Whisper models for Swahili automatic speech recognition.

## Project Overview

This project explores fine-tuning of Whisper models (Tiny and Base variants) on a Swahili speech dataset. The goal is to develop an effective ASR system for Swahili, addressing the challenges of low-resource language speech recognition.

## Dataset

The dataset consists of 5,520 Swahili audio samples with corresponding transcriptions, with the following characteristics:
- Average transcript length: 27.23 words
- Maximum transcript length: 67 words
- Average token length: 71.5 tokens
- Maximum token length: 170 tokens

## Repository Structure

- `whisper-ai-finetuning.ipynb`: Main notebook containing all code for analysis, training, and evaluation
- `README.md`: This file

## Features

The notebook contains several key components:

1. **Data Analysis**
   - Statistical analysis of transcript lengths
   - Token distribution analysis
   - Audio file validation
   - Visualizations of key data characteristics

2. **Model Fine-tuning**
   - Custom dataset implementation for Swahili audio
   - Fine-tuning pipeline for Whisper models
   - Training configuration for both Tiny and Base models
   - Error handling for problematic audio files

3. **Evaluation**
   - Word Error Rate (WER) calculation
   - Character Error Rate (CER) calculation
   - Sample prediction analysis
   - Performance comparison between model variants

## Results

The models were evaluated using Word Error Rate (WER) and Character Error Rate (CER):

- **Whisper Tiny**:
  - WER: 82.95%
  - Training Loss: 0.8426
  - Validation Loss: 1.2317

- **Whisper Base**:
  - WER: 83.41%
  - Training Loss: 0.8244
  - Validation Loss: 1.2615

Despite the models showing high error rates, the project provides valuable insights into the challenges of fine-tuning ASR models for low-resource languages like Swahili.

## Requirements

- transformers
- datasets
- torch
- evaluate
- tensorboard
- jiwer
- librosa
- pandas
- matplotlib
- seaborn

## Usage

1. **Setup Environment**:
   ```bash
   pip install transformers datasets torch evaluate tensorboard jiwer librosa
   ```

2. **Data Preparation**:
   - The code expects audio files in a specific directory structure
   - CSV file with columns for file paths and transcriptions

3. **Run the Analysis**:
   - Execute the data analysis cells to understand dataset characteristics

4. **Fine-tune Models**:
   - Run the training cells to fine-tune Whisper models
   - Multiple model variants can be tested by changing the model name

5. **Evaluate Performance**:
   - Use the evaluation code to assess model performance
   - Test on specific audio samples with the testing code

## Implementation Note

This notebook has been implemented on Kaggle with audio data at path `/kaggle/input/preprocessor/`. The dataset and experiment environment are available on Kaggle. If you want to access the trained model, dataset, or have questions about the implementation, please contact me at zda23m011@iitmz.ac.in and the resources will be shared.

Further experiments are ongoing for Whisper Tiny with larger epoch counts (e.g., 100 epochs), and results will be updated soon.

## Future Work

1. Experiment with larger model variants (Small, Medium)
2. Implement data augmentation techniques
3. Explore curriculum learning approaches
4. Investigate multilingual pre-training benefits

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## Citation

If you use this code in your research, please cite:

```
@inproceedings{
sharma2025finetuning,
title={Fine-tuning Whisper Tiny for Swahili {ASR}: Challenges and Recommendations for Low-Resource Speech Recognition},
author={AVINASH KUMAR SHARMA and Manas R Pandya and Arpit Shukla},
booktitle={6th Workshop on African Natural Language Processing},
year={2025},
url={https://openreview.net/forum?id=QZFoSp5JDL}
}
```
