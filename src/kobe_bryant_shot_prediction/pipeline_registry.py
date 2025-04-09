"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kobe_bryant_shot_prediction.pipelines.PreparacaoDados.pipeline import create_pipeline as preparacao_dados_pipeline
from kobe_bryant_shot_prediction.pipelines.Treinamento.pipeline import create_pipeline as treinamento_pipeline
from kobe_bryant_shot_prediction.pipelines.Aplicacao.pipeline import create_pipeline as aplicacao_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = {
        "PreparacaoDados": preparacao_dados_pipeline(),
        "Treinamento": treinamento_pipeline(),
        "Aplicacao": aplicacao_pipeline(),
    }
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
