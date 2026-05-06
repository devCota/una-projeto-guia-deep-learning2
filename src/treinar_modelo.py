"""
Treinamento de um classificador simples de imagens com fastai e PyTorch.

O projeto usa transfer learning com uma ResNet34 pre-treinada e o dataset
Oxford-IIIT Pets. Para manter o exemplo simples para apresentacao academica,
o modelo aprende a classificar imagens em duas classes: gato e cachorro.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Mantem caches dentro do projeto. Isso ajuda em ambientes com HOME somente
# leitura, como alguns sandboxes, containers e plataformas academicas.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("FASTAI_HOME", str(CACHE_DIR / "fastai"))

import matplotlib
import torch
from fastai.vision.all import (
    CategoryBlock,
    ClassificationInterpretation,
    DataBlock,
    ImageBlock,
    RandomSplitter,
    Resize,
    URLs,
    aug_transforms,
    defaults,
    error_rate,
    get_image_files,
    resnet34,
    untar_data,
    vision_learner,
)

# Backend sem janela grafica. Ajuda quando o script roda em servidor ou Colab.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rotulo_pet(caminho_imagem: Path) -> str:
    """
    Retorna o rotulo da imagem usando a convencao do Oxford-IIIT Pets.

    No dataset original, nomes de arquivos que comecam com letra maiuscula sao
    gatos; nomes que comecam com letra minuscula sao cachorros.
    """
    primeira_letra = caminho_imagem.name[0]
    return "gato" if primeira_letra.isupper() else "cachorro"


def selecionar_imagens(caminho: Path, max_images: int | None = None):
    """
    Lista as imagens do dataset.

    Quando max_images e informado, usa uma amostra balanceada entre gatos e
    cachorros. Essa opcao e util para gerar um modelo de demonstracao rapido.
    """
    imagens = get_image_files(caminho)

    if max_images is None:
        return imagens

    gatos = [imagem for imagem in imagens if rotulo_pet(imagem) == "gato"]
    cachorros = [imagem for imagem in imagens if rotulo_pet(imagem) == "cachorro"]
    por_classe = max(1, max_images // 2)

    return gatos[:por_classe] + cachorros[:por_classe]


def criar_dataloaders(
    caminho_dataset: Path,
    batch_size: int,
    num_workers: int,
    max_images: int | None = None,
):
    """
    Cria os DataLoaders do fastai com separacao treino/validacao.

    item_tfms redimensiona cada imagem individualmente. batch_tfms aplica
    aumentos de dados em lote, como pequenas rotacoes e recortes, para ajudar
    o modelo a generalizar melhor.
    """
    bloco_dados = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=lambda caminho: selecionar_imagens(caminho, max_images),
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=rotulo_pet,
        item_tfms=Resize(460),
        batch_tfms=aug_transforms(size=224, min_scale=0.75),
    )

    return bloco_dados.dataloaders(
        caminho_dataset / "images",
        bs=batch_size,
        num_workers=num_workers,
    )


def treinar_modelo(dls, epochs: int):
    """
    Cria e treina o modelo usando transfer learning.

    vision_learner baixa/carrega uma ResNet34 pre-treinada. O metodo fine_tune
    treina primeiro a cabeca do modelo e depois ajusta parte das camadas da
    rede neural para o nosso problema de classificacao de pets.
    """
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(epochs)
    return learn


def salvar_interpretacao(learn, pasta_relatorios: Path) -> None:
    """
    Salva graficos simples para interpretar o comportamento do modelo.

    A matriz de confusao mostra onde o classificador acerta e erra. O grafico
    de maiores perdas mostra exemplos que confundiram mais o modelo.
    """
    pasta_relatorios.mkdir(parents=True, exist_ok=True)
    interpretacao = ClassificationInterpretation.from_learner(learn)

    interpretacao.plot_confusion_matrix(figsize=(5, 5), dpi=120)
    plt.savefig(pasta_relatorios / "matriz_confusao.png", bbox_inches="tight")
    plt.close()

    interpretacao.plot_top_losses(k=9, nrows=3, figsize=(10, 10))
    plt.savefig(pasta_relatorios / "maiores_erros.png", bbox_inches="tight")
    plt.close()


def exportar_modelo(learn, caminho_modelo: Path) -> None:
    """
    Exporta o modelo treinado para uso posterior na interface grafica.

    O arquivo .pkl contem o pipeline necessario para inferencia: transformacoes,
    vocabulario das classes e pesos treinados.
    """
    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    learn.export(caminho_modelo)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina e exporta um classificador gato/cachorro com fastai."
    )
    parser.add_argument("--epochs", type=int, default=2, help="Numero de epocas.")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Tamanho do lote de treino."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Workers do DataLoader. Use 0 para maior compatibilidade.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/pet_classifier.pkl"),
        help="Arquivo onde o modelo exportado sera salvo.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Pasta onde os graficos de interpretacao serao salvos.",
    )
    parser.add_argument(
        "--skip-interpretation",
        action="store_true",
        help="Pula os graficos de interpretacao para economizar tempo.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Forca o treinamento na CPU, mesmo se houver GPU disponivel.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Usa no maximo N imagens para um treino rapido de demonstracao.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.cpu:
        defaults.device = torch.device("cpu")

    print("Baixando/carregando o Oxford-IIIT Pets...")
    caminho_dataset = untar_data(URLs.PETS, data=PROJECT_ROOT / "data")

    print("Preparando os dados...")
    dls = criar_dataloaders(
        caminho_dataset,
        args.batch_size,
        args.num_workers,
        args.max_images,
    )

    print("Treinando o modelo com transfer learning...")
    learn = treinar_modelo(dls, args.epochs)

    if not args.skip_interpretation:
        print("Gerando graficos de interpretacao...")
        salvar_interpretacao(learn, args.reports_dir)

    print("Exportando o modelo...")
    exportar_modelo(learn, args.model_path)

    print(f"Modelo exportado em: {args.model_path}")
    print("Agora execute: python app.py")


if __name__ == "__main__":
    main()
