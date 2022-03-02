import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def eqaul_species_id(data_path):
    train_df: pd.DataFrame = pd.read_csv(data_path + "train.csv")

    x = train_df.image.values
    y = train_df[["species", "individual_id"]]

    mskf = MultilabelStratifiedKFold(n_splits=5)

    train_df["k_fold"] = -1
    for fold, (train_idxs, val_idxs) in enumerate(mskf.split(x, y)):
        train_df.loc[val_idxs, "k_fold"] = fold

    print(train_df.k_fold.value_counts())

    train_df['cat_id'], id_index = train_df['individual_id'].factorize()


    train_df.to_csv(data_path + "/train_equal_species_ids.csv", index=False)

if __name__ == "__main__":
    data_path = f"{os.path.dirname(os.getcwd())}/data/"
    eqaul_species_id(data_path=data_path)