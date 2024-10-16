# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class DIRCNConfig(ModelConfig):
    num_cascades: int = 30

