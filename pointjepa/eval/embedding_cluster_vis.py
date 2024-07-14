import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.manifold import TSNE
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from pointjepa.datasets.ModelNet40Ply2048 import ModelNet40Ply2048DataModule
from pointjepa.datasets.ShapeNet55 import ShapeNet55DataModule
from pointjepa.datasets.ShapeNetPart import ShapeNetPartDataModule

from pointjepa.models import PointJepa
from pointjepa.utils.checkpoint import extract_model_checkpoint
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def parse_args() -> None:
    parser = argparse.ArgumentParser(
        description="PointJepa Embedding Cluster Visualization"
    )
    parser.add_argument("--finetuned_ckpt_path", "-ch", type=str, required=True)
    return parser.parse_args()


def xy(
    dataloader: DataLoader, model: pl.LightningModule
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_list = []
    label_list = []

    idx =0
    for points, label in tqdm(dataloader):
        # points: torch.Tensor = points.cuda()
        # label: torch.Tensor = label.cuda()
        points = model.val_transformations(points)
        embeddings, centers = model.tokenizer(points.float())
        pos = model.positional_encoding(centers)
        x = model.student(embeddings, pos).last_hidden_state
        x = torch.cat([x.max(dim=1).values, x.mean(dim=1)], dim=-1)
        x_list.append(x.cpu())

        label_list.append(label.cpu())

        # idx = idx +1
        # if idx == 10:
        #     break


    x = torch.cat(x_list, dim=0)  # (N, 768)
    y = torch.cat(label_list, dim=0)  # (N,)
    return x, y


def main():
    #  svm_validation:
    # modelnet40:
    #   class_path: pointjepa.datasets.ModelNet40Ply2048DataModule
    #   init_args:
    #     data_dir: ./data/modelnet40_ply_hdf5_2048
    #     batch_size: 256
    #     drop_last: false
    # cli = CLI(
    #     PointJepa,
    #     trainer_defaults={
    #         "default_root_dir": "artifacts",
    #         # defaults below don't matter here, but will shut up the warnings
    #         "accelerator": "gpu",
    #         "devices": 1,
    #     },
    #     seed_everything_default=0,
    #     save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    #     run=False,
    # )

    # assert isinstance(cli.model, PointJepa)

    li_model = PointJepa()
    args = parse_args()

    li_model = li_model.load_from_checkpoint(args.finetuned_ckpt_path, strict=False)
    # li_model = li_model.cuda()
    li_model.eval()

    data_module = ModelNet40Ply2048DataModule(
        data_dir=str(
            Path(__file__).resolve().parent.parent.parent/ "data/modelnet40_ply_hdf5_2048"
        ),
        batch_size=32,
        drop_last=False,
    )

    data_module.setup()

    x, y = xy(data_module.val_dataloader(), li_model)

    # Dimensionality Reduction with PCA
    # print("Performing PCA...")
    # pca = PCA(n_components=2)
    # x_pca = pca.fit_transform(x.detach().numpy())  # Ensure x is a numpy array

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
    embeddings_tsne = tsne.fit_transform(x.detach().numpy())
    
    label_names = data_module.label_names
    mapped_labels = [label_names[label] for label in y.numpy()]

    # Convert to DataFrame for seaborn plotting
    df = pd.DataFrame({
        'Principal Component 1': embeddings_tsne[:, 0],
        'Principal Component 2': embeddings_tsne[:, 1],
        'Label': mapped_labels  # Ensure y is also a numpy array
    })

    plt.figure(figsize=(14, 8))
    scatter_plot = sns.scatterplot(
    x='Principal Component 1', 
    y='Principal Component 2', 
    hue='Label',  
    data=df, 
     palette=sns.color_palette("hsv", 40)
    )

    scatter_plot.set(xlabel=None, ylabel=None)
    # Remove tick labels and marks
    scatter_plot.set_xticks([])
    scatter_plot.set_yticks([])

    # Position the legend outside the plot area on the right side
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='x-large')

    plt.tight_layout()  # Adjust the layout to make room for the legend
    plt.show()


    # # Convert your DataFrame into one that's suitable for Plotly
    # df = pd.DataFrame({
    #     'Principal Component 1': embeddings_tsne[:, 0],
    #     'Principal Component 2': embeddings_tsne[:, 1],
    #     'Label': mapped_labels
    # })

    # # Create an interactive scatter plot
    # fig = px.scatter(df, x='Principal Component 1', y='Principal Component 2', color='Label', hover_data=['Label'])

    fig.show()
if __name__ == "__main__":
    main()
