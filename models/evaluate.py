from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import torch
from datasets import load_metric

def evaluate_model():
    print("📊 Évaluation du modèle...")
    
    # Charger le modèle
    model = GPT2LMHeadModel.from_pretrained('./models/saved_model')
    tokenizer = GPT2Tokenizer.from_pretrained('./models/saved_model')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Charger les données de test
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Métriques
    bleu = load_metric("bleu")
    
    predictions = []
    references = []
    
    print("🧪 Génération sur le test set...")
    for idx, row in test_df.head(100).iterrows():
        # Extraire le prompt
        prompt = row['full_text'].split('[EMAIL]')[0] + '[EMAIL]'
        
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=300,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if '[EMAIL]' in generated:
            email_content = generated.split('[EMAIL]')[-1].strip()
        else:
            email_content = generated
        
        # Référence réelle
        reference = row['full_text'].split('[EMAIL]')[-1].strip()
        
        predictions.append(email_content)
        references.append([reference])
    
    # Calculer BLEU
    bleu_score = bleu.compute(
        predictions=predictions,
        references=references
    )
    
    print(f"\n✅ Résultats:")
    print(f"   BLEU Score: {bleu_score['bleu']:.4f}")
    
    # Exemples
    print(f"\n📝 Exemples de génération:")
    for i in range(3):
        print(f"\n--- Exemple {i+1} ---")
        print(f"Généré: {predictions[i][:200]}...")

if __name__ == "__main__":
    evaluate_model()
