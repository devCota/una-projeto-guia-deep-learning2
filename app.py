"""
Interface grafica simples com Gradio para usar o modelo exportado.

Antes de executar este arquivo, treine e exporte o modelo com:
python src/treinar_modelo.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Mantem caches locais para funcionar bem em sandboxes, containers e Colab.
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("FASTAI_HOME", str(CACHE_DIR / "fastai"))

import gradio as gr
from fastai.vision.all import load_learner


def carregar_modelo(caminho_modelo: Path):
    """Carrega o arquivo .pkl exportado pelo script de treinamento."""
    if not caminho_modelo.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em '{caminho_modelo}'. "
            "Treine primeiro com: python src/treinar_modelo.py"
        )

    return load_learner(caminho_modelo)


def criar_funcao_predicao(learn):
    """
    Cria a funcao usada pelo Gradio.

    A funcao recebe uma imagem PIL, executa learn.predict e devolve um
    dicionario no formato {classe: confianca}. O componente gr.Label usa esse
    dicionario para mostrar a previsao principal e as probabilidades.
    """

    def predizer(imagem):
        if imagem is None:
            return {}

        _classe_prevista, _indice_classe, probabilidades = learn.predict(imagem)
        classes = learn.dls.vocab

        return {
            str(classes[indice]): float(probabilidades[indice])
            for indice in range(len(classes))
        }

    return predizer


def criar_interface(learn) -> gr.Interface:
    """Monta a interface grafica de inferencia."""
    return gr.Interface(
        fn=criar_funcao_predicao(learn),
        inputs=gr.Image(type="pil", label="Imagem"),
        outputs=gr.Label(num_top_classes=2, label="Previsao"),
        title="Classificador de Pets",
        flagging_mode="never",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa a interface Gradio do classificador de pets."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/pet_classifier.pkl"),
        help="Caminho do modelo exportado pelo fastai.",
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Endereco do servidor. Use 0.0.0.0 no Colab ou em containers.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Porta da interface Gradio.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Cria um link publico temporario do Gradio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    learn = carregar_modelo(args.model_path)
    interface = criar_interface(learn)
    interface.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
