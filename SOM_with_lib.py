import csv
import simpsom as sps
import os
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd
import sompy
import logging

logging.getLogger("matplotlib.font_manager").disabled = True


def parse_dataset_and_train_som_and_create_dir_for_result_image(
    dataset_to_use,
    metric="euclidean",
    sz=20,
    epochs=1000,
    start_learning_rate=0.01,
    neighborhood_function="gaussian",
    topology="rectangular",
    activation_distance="euclidean",
    sigma=10,
    show_emails_per_cell=True,
):
    # check if done already, if so -- skip
    new_fname = f"./{dataset_to_use}_{metric}_sz_{sz}_e_{epochs}_lr_{start_learning_rate}_{neighborhood_function}_{topology}_{activation_distance}_sigma_{sigma}.png"
    if os.path.exists(new_fname):
        print(f"skip {new_fname} as png already exists")
        return

    dataset = []
    labels = []
    paths = []
    # read created csv dataset
    with open(
        f"F:\\_MAGISTRAS\\filtered_datasets\\{dataset_to_use}.csv",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            dataset += [[float(x) for x in row]]

    with open(
        f"F:\\_MAGISTRAS\\filtered_datasets\\{dataset_to_use}_labels.csv",
        encoding="utf-8",
        newline="",
    ) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            labels += ["".join(row)]

    if show_emails_per_cell:
        with open(
            f"F:\\_MAGISTRAS\\filtered_datasets\\{dataset_to_use}_paths.csv",
            encoding="utf-8",
            newline="",
        ) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                paths += ["".join(row)]

    # # minisom
    som = MiniSom(
        sz,
        sz,
        len(dataset[0]),
        sigma=sigma,
        learning_rate=start_learning_rate,
        random_seed=1,
        neighborhood_function=neighborhood_function,
        topology=topology,
        activation_distance=activation_distance,
    )

    som.train(dataset[:], epochs, verbose=True)

    A = dict()

    for i in range(len(dataset)):
        x, y = som.winner(dataset[i])
        if (x, y) not in A:
            A[x, y] = dict(spam=[], phishing=[], ham=[])
        A[x, y][labels[i]] += [paths[i]]

    fig = plt.figure(figsize=(sz, sz))
    the_grid = gridspec.GridSpec(sz, sz, fig)

    labels_map = som.labels_map(dataset, labels)
    label_names = list(set(labels))
    label_names.sort()

    positions = list(labels_map.keys())
    positions.sort()

    for position in positions:
        label_fracs = [labels_map[position][l] for l in label_names]
        fig = plt.subplot(the_grid[sz - 1 - position[1], position[0]], aspect=1)
        patches, texts = plt.pie(label_fracs, wedgeprops={"alpha": 0.75})
        plt.text(
            0,
            0,
            str(sum(labels_map[position].values())),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=7 if sz > 15 else 10,
        )
        # Print 5 emails paths for each class in cell (if available)
        print(position)
        for x in A[position]["ham"][:5]:
            print(f"ham {x}")
        for x in A[position]["spam"][:5]:
            print(f"spam {x}")
        for x in A[position]["phishing"][:5]:
            print(f"phishing {x}")

    plt.legend(
        patches,
        label_names,
        loc="upper right",
        bbox_to_anchor=(-sz / 2, sz / 5),
        ncol=3,
    )
    plt.savefig(f"{new_fname}")

    # Distribution of words per email
    # plt.hist([sum(x) for x in dataset], bins=500, range=[0, 500])


dataset_name = "100000_sz_25_fr_25_w"

for sz in [10, 20]:
    for epochs in [9999]:
        for lr in [0.001]:
            for sigma in [5]:
                for activation_distance in [
                    "euclidean",
                    # "cosine",
                    # "manhattan",
                    # "chebyshev",
                ]:
                    for neighborhood_function in [
                        "gaussian"
                    ]:  # "mexican_hat", "bubble"
                        for topology in ["rectangular"]:  # hexagonal
                            parse_dataset_and_train_som_and_create_dir_for_result_image(
                                dataset_name.split(".")[0],
                                sz=sz,
                                epochs=epochs,
                                start_learning_rate=lr,
                                activation_distance=activation_distance,
                                neighborhood_function=neighborhood_function,
                                topology=topology,
                                sigma=sigma,
                            )

# trying out SOMPY library:
# mapsize = [sz, sz]
# som = sompy.SOMFactory.build(
#     pd.DataFrame(dataset),
#     mapsize,
#     mask=None,
#     mapshape="planar",
#     lattice="rect",
#     normalization="var",
#     initialization="pca",
#     neighborhood="gaussian",
#     training="batch",
#     name="sompy",
# )  # this will use the default parameters, but i can change the initialization and neighborhood methods
# som.train(
#     n_job=1, verbose="info"
# )  # verbose='debug' will print more, and verbose=None wont print anything

# u = sompy.umatrix.UMatrixView(
#     50, 50, "umatrix", show_axis=True, text_size=8, show_text=True
# )

# # This is the Umat value
# UMAT = u.build_u_matrix(som, distance=1, row_normalized=False)

# # Here you have Umatrix plus its render
# UMAT = u.show(
#     som,
#     distance2=1,
#     row_normalized=False,
#     show_data=True,
#     contooor=True,
#     blob=False,
# )

# trying out SimpSOM library:
# Build a network sz x sz with a weights format taken from the raw_data and activate Periodic Boundary Conditions.
# net = sps.SOMNet(sz, sz, dataset, PBC=True, metric=metric, random_seed=0)

# net.train(
#     train_algo="batch",
#     epochs=epochs,
#     start_learning_rate=start_learning_rate,
#     batch_size=1024,
# )

# # cia bus galima pagal zodzius pjuvius paziuret:
# # net.nodes_graph(colnum=0)

# # Project the datapoints on the new 2D network map.
# # cia is esmes ko reikia, tik kad taskeliai vienas kita uzdengia - nelabai matosi kas vyksta
# net.diff_graph()  # cia u matrica nupaiso - tada labelius ant jos galima det
# net.project(dataset, labels=labels)
# # file is named projection_difference.png and its kind of hardcoded - so fixing name
# new_fname = (
#     f"./{dataset_to_use}_{metric}_sz_{sz}_e_{epochs}_lr_{start_learning_rate}.png"
# )
# if os.path.exists(new_fname):
#     os.remove(new_fname)
# os.rename("./projection_difference.png", new_fname)

# Cluster the datapoints according to the Quality Threshold algorithm.
# net.cluster(dataset, clus_type="qthresh")
