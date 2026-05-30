import pandas as pd
import re
import json
from pathlib import Path

def clean_email(text):
    """Nettoie le texte de l'email"""
    if pd.isna(text):
        return ""
    
    # Supprimer les emails forwarded/replied
    text = re.sub(r'-----Original Message-----.*', '', text, flags=re.DOTALL)
    text = re.sub(r'From:.*?Subject:', '', text, flags=re.DOTALL)
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Supprimer les emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def categorize_tone(text):
    """Catégorise le ton de l'email (simple heuristique)"""
    formal_words = ['dear', 'sincerely', 'regards', 'respectfully']
    informal_words = ['hey', 'thanks', 'cheers', 'talk soon']
    
    text_lower = text.lower()
    
    formal_count = sum(1 for word in formal_words if word in text_lower)
    informal_count = sum(1 for word in informal_words if word in text_lower)
    
    if formal_count > informal_count:
        return 'formel'
    elif informal_count > formal_count:
        return 'informel'
    else:
        return 'neutre'

def prepare_dataset():
    print(" Chargement du dataset brut...")
    df = pd.read_csv('data/raw/emails.csv')
    
    print(f" {len(df)} emails chargés")
    print(" Nettoyage en cours...")
    
    # Garder seulement message
    if 'message' in df.columns:
        df = df[['message']].copy()
    elif 'content' in df.columns:
        df = df[['content']].copy()
        df.rename(columns={'content': 'message'}, inplace=True)
    
    # Nettoyer
    df['message'] = df['message'].apply(clean_email)
    
    # Filtrer les emails trop courts/longs
    df['length'] = df['message'].str.len()
    df = df[(df['length'] > 50) & (df['length'] < 2000)]
    
    # Catégoriser le ton
    print(" Catégorisation du ton...")
    df['tone'] = df['message'].apply(categorize_tone)
    
    # Créer des contextes synthétiques (simplifié)
    contexts = [
        "Proposer un rendez-vous",
        "Demander des informations",
        "Répondre à une question",
        "Faire un suivi",
        "Remercier",
        "Partager des informations"
    ]
    
    df['context'] = [contexts[i % len(contexts)] for i in range(len(df))]
    
    # Format pour le fine-tuning
    df['input_text'] = df.apply(
        lambda x: f"[CONTEXT] {x['context']} [TONE] {x['tone']} [EMAIL]",
        axis=1
    )
    
    df['full_text'] = df['input_text'] + " " + df['message']
    
    # Split train/val/test
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size+val_size]
    test_df = df[train_size+val_size:]
    
    # Sauvegarder
    Path('data/processed').mkdir(exist_ok=True)
    train_df[['full_text']].to_csv('data/processed/train.csv', index=False)
    val_df[['full_text']].to_csv('data/processed/val.csv', index=False)
    test_df[['full_text']].to_csv('data/processed/test.csv', index=False)
    
    print(f" Dataset préparé:")
    print(f"   - Train: {len(train_df)} emails")
    print(f"   - Validation: {len(val_df)} emails")
    print(f"   - Test: {len(test_df)} emails")
    
    # Statistiques
    print(f"\n Statistiques:")
    print(df['tone'].value_counts())
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_dataset()
