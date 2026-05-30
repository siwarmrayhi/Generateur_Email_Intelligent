"""
Script d'entraînement - Version Colab
"""

import torch
import os
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import pandas as pd
from datetime import datetime

def train_model():
    print("="*70)
    print("ENTRAÎNEMENT DU MODÈLE - EMAIL AI GENERATOR")
    print("="*70)
    
    # Configuration
    project_path = '/content/drive/MyDrive/projet_ia'
    model_name = 'gpt2'
    
    # Vérifier GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    
    if device == "cpu":
        print("⚠️  WARNING: GPU non disponible!")
        return
    
    # Charger le modèle
    print("\n Chargement de GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    print(f" Modèle chargé: {model.num_parameters():,} paramètres")
    
    # Charger les données (MÉTHODE CORRIGÉE)
    print("\n Chargement des données...")
    
    train_path = f'{project_path}/data/processed/train.csv'
    val_path = f'{project_path}/data/processed/val.csv'
    
    # Vérifier l'existence des fichiers
    import os
    if not os.path.exists(train_path):
        print(f" Fichier non trouvé: {train_path}")
        return
    if not os.path.exists(val_path):
        print(f" Fichier non trouvé: {val_path}")
        return
    
    # Charger avec pandas puis convertir en Dataset
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    dataset = {
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df)
    }
    
    print(f" Train: {len(dataset['train'])} exemples")
    print(f" Validation: {len(dataset['validation'])} exemples")
    
    # Tokenization
    print("\n Tokenization...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['full_text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )
    
    tokenized_dataset = {
        'train': dataset['train'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing train"
        ),
        'validation': dataset['validation'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['validation'].column_names,
            desc="Tokenizing validation"
        )
    }
    
    print(" Tokenization terminée")
    
    # Configuration de l'entraînement
    print("\n  Configuration de l'entraînement...")
    
    output_dir = f'{project_path}/models/email_generator'
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{project_path}/logs',
        logging_steps=100,
        eval_steps=500,
        save_steps=1000,
        evaluation_strategy="steps",
        save_total_limit=2,
        learning_rate=5e-5,
        load_best_model_at_end=True,
        fp16=True,
        report_to="none",
        optim="adamw_torch",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
    )
    
    # ENTRAÎNEMENT
    print("\n" + "="*70)
    print("  DÉMARRAGE DE L'ENTRAÎNEMENT")
    print("="*70)
    print("⏱  Durée estimée: 1-3 heures")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    
    try:
        train_result = trainer.train()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Résultats
        print("\n" + "="*70)
        print(" ENTRAÎNEMENT TERMINÉ!")
        print("="*70)
        print(f"  Durée: {duration}")
        print(f" Loss finale: {train_result.training_loss:.4f}")
        
        # Sauvegarder
        print("\n Sauvegarde du modèle...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f" Modèle sauvegardé dans: {output_dir}")
        
        # Évaluation
        print("\n Évaluation finale...")
        eval_results = trainer.evaluate()
        print(f" Eval Loss: {eval_results['eval_loss']:.4f}")
        print(f" Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")
        
        print("\n" + "="*70)
        print(" TOUT EST TERMINÉ!")
        print("="*70)
        
    except Exception as e:
        print(f"\n Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model()
