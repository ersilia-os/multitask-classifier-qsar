import os
import shutil
import tempfile
import json
from typing import List
from loguru import logger
import pandas as pd


class SingleOutputClassifier(object):
    def __init__(self, dir: str, name: str) -> None:
        self.name = name
        self.dir = os.path.abspath(dir)
        if not os.path.exists(self.dir):
            raise FileNotFoundError("Folder {0} does not exist".format(self.dir))

    def fit(self, df: pd.DataFrame) -> None:
        smiles_list = df["smiles"]
        y = df[self.name]

    def predict(self) -> None:
        pass

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass


class AutoClassifier(object):
    def __init__(self, dir: str) -> None:
        self.dir = os.path.abspath(dir)
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.mkdir(self.dir)

    def fit(self, df: pd.DataFrame) -> None:
        columns = list(df.columns)[1:]
        for col in columns:
            logger.info("Working on task: {0}".format(col))
            df_ = df[~df[col].isnull()]
            clf = SingleOutputClassifier(dir=self.dir, name=col)
            clf.fit(df_)
            clf.save()
        self.columns = columns
        with open(os.path.join(self.dir, "column_names.json"), "w") as f:
            json.dump(self.columns, f, indent=4)
    
    def predict(self, smiles_list: List[str]) -> None:
        R = []
        for col in self.columns:
            logger.info("Loading classifier: {0}".format(col))
            clf = SingleOutputClassifier(dir=self.dir, name=col)
            clf.load()
            

    def load(self) -> None:
        with open(os.path.join(self.dir, "column_names.json"), "r") as f:
            self.columns = json.load()

