import os
import shutil
import tempfile
import json
from typing import List
from loguru import logger
import pandas as pd

from qsartuna.three_step_opt_build_merge import (
    optimize,
    buildconfig_best,
    build_best,
    build_merged,
)
from qsartuna.config import ModelMode, OptimizationDirection
from qsartuna.config.optconfig import (
    OptimizationConfig,
    ChemPropHyperoptClassifier,
    XGBClassifier,
)
from qsartuna.datareader import Dataset
from qsartuna.descriptors import ECFP_counts


class SingleOutputClassifier(object):
    def __init__(self, dir: str, name: str) -> None:
        self.name = name
        self.dir = os.path.abspath(dir)
        if not os.path.exists(self.dir):
            raise FileNotFoundError("Folder {0} does not exist".format(self.dir))
        
    def _prepare_optimization_config(self, train_csv) -> OptimizationConfig:
        logger.info("Preparing hyperparameter optimization configuration")
        config = OptimizationConfig(
            data=Dataset(
                input_column="smiles",
                response_column="y",
                training_dataset_file=train_csv,
            ),
            descriptors=[ECFP_counts.new()],
            algorithms=[
                ChemPropHyperoptClassifier.new(),
                XGBClassifier.new(),
            ],
            settings=OptimizationConfig.Settings(
                mode=ModelMode.CLASSIFICATION,
                cross_validation=3,
                n_trials=100,
                direction=OptimizationDirection.MAXIMIZATION,
            ),
        )
        return config

    def _run_optuna_train(self, config: OptimizationConfig) -> None:
        logger.info("Starting hyperparameter optimization")
        study = optimize(config, study_name=self.name)
        logger.info("Hyperparameter optimization finished")
        logger.info("Getting best trial config")
        buildconfig = buildconfig_best(study)
        with open(os.path.join(self.dir, "best_config.txt"), "w") as f:
            f.write(str(buildconfig.__dict__))
        logger.info("Building (re-training) and save the best model")
        build_best(buildconfig, os.path.join(self.dir, "target/best.pkl"))
        logger.info("Building and saving the model on the merged train+test data")
        build_merged(buildconfig, os.path.join(self.dir, "target/merged.pkl"))

    def fit(self, df: pd.DataFrame) -> None:
        logger.info("Fitting model")
        smiles_list = df["smiles"]
        y = df[self.name]
        train_csv = os.path.join(self.dir, "train.csv")
        with open(train_csv, "w") as f:
            f.write("smiles,y\n")
            for i in range(len(smiles_list)):
                f.write("{0},{1}\n".format(smiles_list[i], y[i]))
        config = self._prepare_optimization_config(train_csv)
        self._run_optuna_train(config)
        logger.info("Done with training! Check your {0} folder".format(self.dir))
    
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

