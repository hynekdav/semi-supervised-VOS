# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop
from abc import ABC, abstractmethod


class BaseTripletMiner(ABC):

    @abstractmethod
    def get_triplets(self, embeddings, labels):
        pass
