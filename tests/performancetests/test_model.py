#script to test model that is being staged
import wandb
import os
import torch
from src.arginator_protein_classifier.model import Lightning_Model

def load_model(model_name):#originally said load_model(artifact)
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_name) #originally was api.artifact(model_checkpoint)
    
    if "binary" in model_name:
        problem = "binary/"
    else:
        problem = "multiclass/"

    logdir = ".models/" + problem
    #os.makedirs(logdir, exist_ok=True)
    artifact.download(root=logdir)
    file_name = artifact.files()[0].name
    return Lightning_Model.load_from_checkpoint(f"{logdir}/{file_name}", weights_only=False)

def test_forward_pass():
    model = load_model(os.getenv("MODEL_NAME"))
    model.eval()
    #model= Lightning_Model.load_from_checkpoint(".models/binary/model.ckpt", weights_only=False)

    x = torch.randn(4, 1024)  # batch of mock mean-pooled ProtT5 embeddings
    with torch.no_grad():
        y = model(x)

    assert y.shape == torch.rand(4, model.output_dim).shape
    # Check output shape

