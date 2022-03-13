import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch import rand


def eqaul_species_id(data_path):
    df: pd.DataFrame = pd.read_csv(data_path + "train.csv")

    x = df.image.values
    y = df[["species", "individual_id"]]

    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=0)

    df["k_fold"] = -1
    for fold, (train_idxs, val_idxs) in enumerate(mskf.split(x, y)):
        df.loc[val_idxs, "k_fold"] = fold

    print(df.k_fold.value_counts())

    df["cat_id"], id_index = df["individual_id"].factorize()

    # renaming later after the split as we are both merging couple and just fixing typo in couple. The disctribution is not that different either way.
    # the training itself if based on id and not species so old runs are ok.
    # df["species"].replace(
    #     {
    #         "bottlenose_dolpin": "bottlenose_dolphin",
    #         "kiler_whale": "killer_whale",
    #         "pilot_whale": "short_finned_pilot_whale",
    #         "globis": "short_finned_pilot_whale",
    #     },
    #     inplace=True,
    # )

    df.to_csv(data_path + "/train_equal_species_ids.csv", index=False)


if __name__ == "__main__":
    data_path = f"{os.path.dirname(os.getcwd())}/data/"
    eqaul_species_id(data_path=data_path)
