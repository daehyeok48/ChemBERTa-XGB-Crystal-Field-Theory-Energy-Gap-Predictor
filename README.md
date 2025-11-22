# ChemBERTa-XGB Crystal Field Theory - Energy Gap Predictor  
Author: Daehyeok Kim @ KENTECH (kimdaehyeok@kentech.ac.kr)
Date: Nov 24, 2025
Author Github: https://github.com/daehyeok48/ChemBERTa-XGB-Crystal-Field-Theory-Energy-Gap-Predictor

A full pipeline for predicting molecular Energy Gap using a hybrid feature model:
- Local ChemBERTa transformer embeddings  
- Morgan Fingerprints (2048-bit)  
- MACCS Keys  
- RDKit 2D Descriptors  
- XGBoost regression  
- K-Fold cross-validation  
- SMILES augmentation  
- Molecule 2D PNG generation  

## Project Structure
CFT_Energy_predictor/
│
├── app/
│   └── app.py
│
├── execute/
│   ├── train.py
│   └── infer.py
│
├── src/
│   ├── encoder.py
│   ├── fingerprints.py
│   ├── augment.py
│   ├── features.py
│   ├── model.py
│   ├── predict.py
│   ├── visualize.py
│   └── utils.py
│
├── models/
│   ├── chemberta/
│   ├── scaler.pkl
│   └── chemberta_trained.pkl
│
├── data/
│   └── smiles_dataset.csv
│
├── molecule_images/
│
├── environment.yml
│
└── README.md

## Setup
Require conda environment in your local setting
- conda env create -f environment.yml
- conda activate CFT_Energy_predictor

## Training
If you need additional training with your dataset use code below:
- python -m execute.train
(Re-training is not necessary)

This will:
- Augment SMILES  
- Extract all features  
- Run 5-fold cross validation  
- Train final model  
- Save:
  - `models/scaler.pkl`
  - `models/chemberta_trained.pkl`

## Inference (Single Molecule)
- python -m execute.infer

Output:
- Predicted Energy Gap  
- PNG saved at `molecule_images/`  

## User CLI App
- python -m app.CFT_Energy_predictor

## Acknowledgement
This project is for EF2039 Term project 1 assignment.
ChemBERTa source are retrieved from Chithrananda, S., Grand, G., & Ramsundar, B. (2020). ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. ArXiv, abs/2010.09885.
Author Hugging Face: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/
