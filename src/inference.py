import torch
from dataset import Whales
# from pytorch_metric_learning.utils.inference import FaissKNN
import faiss
# from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


def inference(opt, model, trainer):

    reference_loader, query_loader = get_loaders(opt)

    outputs = trainer.predict(model, reference_loader)
    reference_embeddings = torch.cat([batch_out["embeddings"] for batch_out in outputs])
    reference_labels = torch.cat([batch_out["labels"] for batch_out in outputs])
    # reference_embeddings = reference_embeddings[:3]  # debug

    outputs = trainer.predict(model, query_loader)
    query_embeddings = torch.cat([batch_out["embeddings"] for batch_out in outputs])
    # query_embeddings = query_embeddings[:2]  # debug
    del outputs

    torch.save(reference_embeddings, "reference_embeddings.pt")
    torch.save(reference_labels, "reference_labels.pt")
    torch.save(query_embeddings, "query_embeddings.pt")

    # knn_func = FaissKNN(index_init_fn=faiss.IndexFlatIP)
    # distances, indices = knn_func(
    #     query_embeddings, opt.k_nn, reference_embeddings, False
    # )

    # print("indices", indices.shape)
    # print("distances", distances.shape)

    # acc_calc = AccuracyCalculator(
    #     include=(),
    #     exclude=(),
    #     avg_of_avgs=False,
    #     return_per_class=False,
    #     k=None,
    #     label_comparison_fn=None,
    #     device=None,
    #     knn_func=None,
    #     kmeans_func=None,
    # )


def get_loaders(opt):
    def get_loader(opt, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=opt.pin_memory,
            shuffle=False,
        )

    if opt.run_type == "cv":
        reference_loader = get_loader(
            opt,
            Whales(
                folds=[0, 1, 2, 3],
                no_augment=True,
                data_root=opt.data_root,
                image_path=opt.train_image_path,
                csv_path=opt.train_csv_path,
            ),
        )

        query_loader = get_loader(
            opt,
            Whales(
                folds=[4],
                no_augment=True,
                data_root=opt.data_root,
                image_path=opt.train_image_path,
                csv_path=opt.train_csv_path,
            ),
        )

    elif opt.run_type == "test":
        reference_loader = get_loader(
            opt,
            Whales(
                folds=[0, 1, 2, 3, 4],
                no_augment=True,
                data_root=opt.data_root,
                image_path=opt.train_image_path,
                csv_path=opt.train_csv_path,
            ),
        )

        query_loader = get_loader(
            opt,
            Whales(
                folds=[],
                no_augment=True,
                data_root=opt.data_root,
                image_path=opt.test_image_path,
                csv_path=opt.test_csv_path,
            ),
        )
    else:
        raise NotImplementedError

    return reference_loader, query_loader
