import os
import time

import h5py
import torch
from huggingface_hub import hf_hub_download
from transformers import T5EncoderModel, T5Tokenizer

# Check device once on import
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Convert.py using device: {device}")

def load_t5_model(model_dir=None, transformer_link="Rostlab/prot_t5_xl_half_uniref50-enc"):
    """
    Loads the heavy model into memory. Call this ONLY ONCE in the backend startup.
    """
    # Force strings immediately
    transformer_link = str(transformer_link)
    if model_dir:
        model_dir = str(model_dir)

    print(f"DEBUG: Loading model from: {transformer_link}")
    
    # 1. Load the Model
    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)
    
    # Cast to full precision if on CPU
    if device.type == "cpu":
        print("Casting model to full precision for CPU...")
        model.to(torch.float32)
    
    model = model.to(device)
    model = model.eval()
    
    # 2. Load the Tokenizer (The "Manual" Way)
    print("DEBUG: Resolving tokenizer file manually...")
    try:
        # Explicitly download/find the spiece.model file
        # This returns the absolute path as a string, bypassing the bug
        vocab_path = hf_hub_download(
            repo_id=transformer_link, 
            filename="spiece.model", 
            cache_dir=model_dir
        )
        print(f"DEBUG: Vocab file resolved to: {vocab_path}")
        
        # Load tokenizer using the direct file path
        vocab = T5Tokenizer.from_pretrained(str(vocab_path), do_lower_case=False, legacy=True)
        
    except Exception as e:
        print(f"CRITICAL ERROR loading tokenizer: {e}")
        # If manual download fails, fallback to standard (unlikely to work if above failed, but good safety)
        vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, legacy=True)

    return model, vocab

def read_fasta(fasta_path):
    sequences = dict()
    with open(fasta_path, 'r') as fasta_f:
        uniprot_id = None
        for line in fasta_f:
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip()
                # sanitize ID
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                sequences[uniprot_id] = ''
            elif uniprot_id:
                # Append sequence, remove whitespace/gaps, upper case
                sequences[uniprot_id] += ''.join(line.split()).upper().replace("-","")
    return sequences

# CHANGE 1: Added 'callback=None' to arguments
def run_conversion(seq_path, emb_path, model, vocab, per_protein=True, max_residues=4000, max_seq_len=1000, max_batch=100, callback=None):
    """
    Main logic to convert .fa to .h5 using the pre-loaded model.
    """
    seq_dict_raw = read_fasta(seq_path)
    emb_dict = dict()

    # Sort sequences by length (efficiency optimization)
    # seq_dict becomes a list of tuples: [('id1', 'SEQ...'), ('id2', 'SEQ...')]
    seq_dict = sorted(seq_dict_raw.items(), key=lambda kv: len(kv[1]), reverse=True)
    
    # CHANGE 2: Calculate total for progress tracking
    total_seqs = len(seq_dict)
    
    batch = list()
    
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        # Replace non-standard amino acids with X
        seq = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_len = len(seq)
        seq_spaced = ' '.join(list(seq)) # T5 requires spaces between residues
        batch.append((pdb_id, seq_spaced, seq_len))

        # Check batch limits
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        
        # If batch is full or it's the last sequence, process it
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == total_seqs or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # Tokenize
            token_encoding = vocab.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                continue

            # Extract embeddings
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]

                if per_protein:
                    emb = emb.mean(dim=0) # Mean pooling

                emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()
        
        # CHANGE 3: Report progress back to main.py
        # We place this here so the bar updates as we iterate through sequences
        if callback:
            callback(seq_idx, total_seqs)

    # Save to H5
    with h5py.File(str(emb_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            hf.create_dataset(sequence_id, data=embedding)
    
    return True