# DVC Commands
dvc stage add -n preprocess -d src/preprocess.py -d data/raw -d params.yaml -o data/processed python src/preprocess.py

dvc stage add -n train -d src/train.py -d src/model.py -d data/processed -d params.yaml -o model.pt -M metrics.json python src/train.py
