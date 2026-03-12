
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
from KNN import *
from Kmeans import *


# FLAGS DE EJECUCIÓN
RUN_MAIN_DEMO = True
RUN_BENCHMARKS = False
RUN_DECAY_ANALYSIS = False
RUN_DECAY_VISUALIZATION = False


### Funcions d'analisi qualitatiu

def Retrieval_by_color(llista_imatges, etiquetes, cerca_colors, visualize=True):
    """
    Funció que rep com entrada una llista d'imatges, les etiquetes que hem obtingut en
    aplicar l'algorisme Kmeans a aquestes imatges i la pregunta que fem per a una cerca
    concreta (això és, un string o una llista d'strings amb els colors que volem buscar).
    Retorna totes les imatges que contenen les etiquetes de la pregunta que fem. Aquesta
    funció pot ser millorada afegint un paràmetre d'entrada que contingui el percentatge
    de cada color que conté la imatge, i retorni les imatges ordenades segons el
    percentatge.
    """
    if isinstance(cerca_colors, str):
        cerca_colors = [color.strip() for color in cerca_colors.split(',')]
    else:
        cerca_colors = [color.strip() for color in cerca_colors]

    resultats = []

    for i, color_percent_list in enumerate(etiquetes):
        match_score = 0.0
        for color, percent in color_percent_list:
            if color in cerca_colors:
                match_score += percent
        if match_score > 0:
            resultats.append((match_score, llista_imatges[i], color_percent_list, i))

    resultats.sort(reverse=True, key=lambda x: x[0])

    imatges_resultat = [r[1] for r in resultats]
    info_resultat = [
        ", ".join(f"{color} ({int(p * 100)}%)" for color, p in r[2])
        for r in resultats
    ]
    title = f"Cerca per color (ordenada) - Colors buscats: {', '.join(cerca_colors)}"

    if visualize:
        visualize_retrieval(
            imatges_resultat,
            len(imatges_resultat),
            title=title,
            info=info_resultat
        )
    return resultats


def Retrieval_by_shape(llista_imatges, etiquetes_knn, neighbours_knn, cerca_formes, visualize=True):
    """
    Funció que rep com entrada una llista d'imatges, les etiquetes que hem obtingut en
    aplicar l'algorisme KNN a aquestes imatges i la pregunta que fem per a una cerca
    concreta (això és, un string definint la forma de roba que volem buscar). Retorna
    totes les imatges que contenen l'etiqueta de la pregunta que fem. Aquesta funció pot
    ser millorada afegint un paràmetre d'entrada que contingui el percentatge de
    K-neighbors amb l'etiqueta que busquem i retorni les imatges ordenades segons el
    percentatge.
    """
    resultats = []

    for i, neighbours in enumerate(neighbours_knn):
        if etiquetes_knn[i] != cerca_formes:
            continue
        total = len(neighbours)
        match = np.sum(np.array(neighbours) == cerca_formes)
        percentatge = match / total if total > 0 else 0

        if percentatge > 0:
            resultats.append((percentatge, llista_imatges[i], neighbours_knn[i], i))

    resultats.sort(reverse=True, key=lambda x: x[0])

    imatges_resultat = [r[1] for r in resultats]
    info_resultat = [f"{int(r[0] * 100)}% - {r[2]}" for r in resultats]
    title = f"Cerca per forma (ordenada) - Forma buscada: {cerca_formes}"

    if visualize:
        visualize_retrieval(
            imatges_resultat,
            len(imatges_resultat),
            title=title,
            info=info_resultat
        )

    return resultats


def Retrieval_combined(llista_imatges, etiquetes_color, etiquetes_forma, neighbours_knn,
                       cerca_colors, cerca_formes, visualize=True):
    """
    Funció que rep com a entrada una llista d'imatges, les etiquetes de forma i les de
    color, una pregunta de forma i una pregunta de color. Retorna les imatges que
    coincideixen amb les dues preguntes, per exemple: Red Flip Flops.
    """
    color_resultats = Retrieval_by_color(llista_imatges, etiquetes_color, cerca_colors, visualize=False)
    forma_resultats = Retrieval_by_shape(llista_imatges, etiquetes_forma, neighbours_knn, cerca_formes, visualize=False)

    color_dict = {r[3]: (r[0], r[2]) for r in color_resultats}
    forma_dict = {r[3]: (r[0], r[2]) for r in forma_resultats}

    imatges_comunes = set(color_dict.keys()) & set(forma_dict.keys())

    resultats_comb = []
    for img in imatges_comunes:
        color_score, color_info = color_dict[img]
        forma_score, kn_neighbours = forma_dict[img]
        combined_score = (color_score + forma_score) / 2
        resultats_comb.append((combined_score, llista_imatges[img], color_info, forma_score))

    resultats_comb.sort(reverse=True, key=lambda x: x[0])

    imatges_resultat = [r[1] for r in resultats_comb]
    info_resultat = [
        f"Colors: {', '.join(f'{color} ({int(p * 100)}%)' for color, p in r[2])} | "
        f"Forma: {int(r[3] * 100)}%"
        for r in resultats_comb
    ]
    title = f"Cerca combinada - Colors: {cerca_colors} | Forma: {cerca_formes}"

    if visualize:
        visualize_retrieval(
            imatges_resultat,
            len(imatges_resultat),
            title=title,
            info=info_resultat
        )

    return resultats_comb


### Funcions d'analisi quantitatiu

def Kmean_statistics(kmeans, llista_imatges, kmax):
    """
    Funció que rep com a entrada la classe Kmeans amb un conjunt d'imatges i un valor,
    Kmax, que representa la màxima K que volem analitzar. Per cada valor des de K=2 fins a
    K=Kmax executarà la funció fit i calcularà la WCD, el nombre d'iteracions i el temps
    que ha necessitat per convergir, etc. Finalment, farà una visualització amb aquestes
    dades.
    """
    wcds = []
    iterations = []
    times = []

    Ks = list(range(2, kmax + 1))
    for image in llista_imatges:
        for K in Ks:
            kmeans = KMeans(image, K)
            start_time = time.time()
            kmeans.fit()

            times.append(time.time() - start_time)
            wcds.append(kmeans.withinClassDistance())
            iterations.append(kmeans.num_iter)

        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(Ks, wcds, marker='o')
        plt.title('Within Cluster Distance (WCD)')
        plt.xlabel('K')
        plt.ylabel('WCD')

        plt.subplot(1, 3, 2)
        plt.plot(Ks, times, marker='o', color='orange')
        plt.title('Temps de convergència')
        plt.xlabel('K')
        plt.ylabel('Segons')

        plt.subplot(1, 3, 3)
        plt.plot(Ks, iterations, marker='o', color='green')
        plt.title('Iteracions fins a la convergència')
        plt.xlabel('K')
        plt.ylabel('Iteracions')

        plt.suptitle('Estadístiques del KMeans segons K')
        plt.tight_layout()
        plt.show()


def Get_shape_accuracy(llista_etiquetes_knn, ground_truth):
    """
    Funció que rep com a entrada les etiquetes que hem obtingut en aplicar el KNN i el
    Ground-Truth d'aquestes. Retorna el percentatge d'etiquetes correctes.
    """
    iguals = 0

    for etiqueta, ground in zip(llista_etiquetes_knn, ground_truth):
        if etiqueta == ground:
            iguals += 1
    return ((iguals / len(llista_etiquetes_knn)) * 100)


def Get_color_accuracy(llista_etiquetes_kmeans, ground_truth):
    """
    Funció que rep com a entrada les etiquetes que hem obtingut en aplicar el kmeans i el
    Ground-Truth d'aquestes. Retorna el percentatge d'etiquetes correctes.
    """
    total_similitud = 0.0
    n = len(llista_etiquetes_kmeans)

    for pred, gt in zip(llista_etiquetes_kmeans, ground_truth):
        pred_set = set(pred)
        gt_set = set(gt)
        if not pred_set and not gt_set:
            similitud = 1.0
        elif not pred_set or not gt_set:
            similitud = 0.0
        else:
            interseccio = pred_set & gt_set
            unio = pred_set | gt_set
            similitud = len(interseccio) / len(unio)

        total_similitud += similitud

    return round((total_similitud / n) * 100, 2)


def benchmark_kmeans_initializations(
    images_to_use,
    ground_truth_colors,
    clusterer_class,
    kmax=10,
    iters=20,
    min_per=0.0,
    decay_threshold=0.1
):
    methods = ['first', 'diagonal', 'random', 'kmeans++']
    scores = {method: [] for method in methods}
    k = 3
    print(f"\nStarting benchmark with {iters} iterations per method...\n")

    for method in methods:
        print(f"Method: {method}")
        for i in range(iters):
            predicted_colors = []

            for image in images_to_use:
                kmeans = clusterer_class(
                    image,
                    k,
                    options={
                        'fitting': 'Fisher',
                        'decay_threshold': decay_threshold,
                        'km_init': method,
                        'filter_border': True,
                        'filter_tolerance': 22
                    }
                )
                kmeans.find_bestK(kmax)
                kmeans.fit()

                colors = kmeans.get_dominant_colors(min_per)
                predicted_colors.append(list(set([color for color, _ in colors])))

            acc = Get_color_accuracy(predicted_colors, ground_truth_colors)
            print(f"  Iteration {i+1}/{iters} accuracy: {acc:.2f}%")
            scores[method].append(acc)

            if method == 'first' or method == 'diagonal':
                break

        avg = np.mean(scores[method])
        print(f"Avg accuracy for '{method}': {avg:.2f}%\n")

    avg_scores = {m: np.mean(scores[m]) for m in methods}
    best_score = max(avg_scores.values())
    best_methods = [m for m, s in avg_scores.items() if s == best_score]

    print(f"\nBest-performing method(s): {', '.join(best_methods)} with {best_score:.2f}% accuracy\n")

    plt.figure(figsize=(8, 5))
    plt.bar(avg_scores.keys(), avg_scores.values(), color='cornflowerblue')
    plt.ylabel("Accuracy (%)")
    plt.title(f"Benchmark of K-Means Init Methods (kmax={kmax}, iters={iters})")
    plt.ylim(40, 42)
    plt.grid(axis='y')
    plt.show()

    return avg_scores


def resize_calculate_features(images_array, new_size):
    features_r = []
    features_g = []
    features_b = []

    for image in images_array:
        resized_image = downscale_images(image, new_size)

        red_channel = resized_image[:, :, 0]
        green_channel = resized_image[:, :, 1]
        blue_channel = resized_image[:, :, 2]

        h, w = red_channel.shape
        features_r.append([np.mean(red_channel), np.var(red_channel), np.mean(red_channel[:h // 2, :]) / np.mean(red_channel[h // 2:, :])])
        features_g.append([np.mean(green_channel), np.var(green_channel), np.mean(green_channel[:h // 2, :]) / np.mean(green_channel[h // 2:, :])])
        features_b.append([np.mean(blue_channel), np.var(blue_channel), np.mean(blue_channel[:h // 2, :]) / np.mean(blue_channel[h // 2:, :])])

    return np.stack([features_r, features_g, features_b], axis=2)


def downscale_images(llista_imatges, new_size):
    if llista_imatges.ndim > 3:
        old_height, old_width = llista_imatges.shape[1:3]
    else:
        old_height, old_width = llista_imatges.shape[:2]

    new_width, new_height = new_size

    row_scale = old_height / new_height
    col_scale = old_width / new_width

    row_idx = (np.arange(new_height) * row_scale).astype(int)
    col_idx = (np.arange(new_width) * col_scale).astype(int)

    if llista_imatges.ndim > 3:
        return np.array([image[row_idx[:, None], col_idx, :] for image in llista_imatges])
    else:
        return np.array(llista_imatges[row_idx[:, None], col_idx, :])


def get_crop_window(imatge):
    white_pixel = np.array([255, 255, 255])

    for i, row_pixels in enumerate(imatge):
        for j, col_pixel in enumerate(row_pixels):
            diff = np.abs(white_pixel - col_pixel)
            if np.all(diff <= 5):
                upper = (i, j)
                return upper

    return (0, 0)


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    images_dir = base_dir / 'images'
    gt_json = images_dir / 'gt.json'
    gt_reduced_json = images_dir / 'gt_reduced.json'

    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(
            root_folder=images_dir,
            gt_json=gt_json
        )
    print(train_imgs.shape)

    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset(
        root_folder=images_dir,
        extended_gt_json=gt_reduced_json
    )
    cropped_images = crop_images(imgs, upper, lower)

    if RUN_MAIN_DEMO:
        ntest = 30
        k = 3
        images_to_test = imgs[:ntest]
        labels_to_test = class_labels[:ntest]

        kmax = 10
        colors2 = []
        colors_and_percentage = []
        color_res = color_labels[:ntest]

        for image in images_to_test:
            kmeans = KMeans(
                image,
                5,
                options={'fitting': 'Fisher', 'decay_threshold': 0.1, 'km_init': 'diagonal',
                         'filter_border': True, 'filter_tolerance': 22}
            )
            kmeans.find_bestK(kmax)
            kmeans.fit()
            dominant_colors = kmeans.get_dominant_colors(0.0)
            colors_and_percentage.append(dominant_colors)
            colors2.append(list(set([color for color, _ in dominant_colors])))

        accuracy = Get_color_accuracy(colors2, color_res)
        print("Percentatge d'etiquetes correctes per a Kmeans:", accuracy)

        ntest = 180
        colors2 = []
        colors_and_percentage = []
        kmax = 10
        min_per = 0.0
        images_to_use = imgs[:ntest]
        cropped_images_to_use = cropped_images[:ntest]
        real_shapes = class_labels[:ntest]
        resize = (30, 40)

        resized_images_to_use = downscale_images(images_to_use, resize)
        resized_train_imgs = downscale_images(train_imgs, resize)

        knn = KNN(resized_train_imgs, train_class_labels)
        pred = knn.predict(resized_images_to_use, 3)
        accuracy = Get_shape_accuracy(pred, real_shapes)
        print("Percentatge d'etiquetes correctes per a KNN:", accuracy)

        for image in images_to_use:
            kmeans = KMeans(
                image,
                5,
                options={'fitting': 'Fisher', 'decay_threshold': 0.1, 'km_init': 'diagonal',
                         'filter_border': True, 'filter_tolerance': 22}
            )
            kmeans.find_bestK(kmax)
            kmeans.fit()
            dominant_colors = kmeans.get_dominant_colors(min_per)
            colors_and_percentage.append(dominant_colors)
            colors2.append(list(set([color for color, _ in dominant_colors])))

        Retrieval_by_shape(imgs[:ntest], pred, knn.neighbors, 'Shorts')
        Retrieval_combined(images_to_use, colors_and_percentage, pred, knn.neighbors, 'Black', 'Shorts')
        Retrieval_by_color(imgs[:ntest], colors_and_percentage, 'Pink')

    if RUN_BENCHMARKS:
        ntest = 180
        colors2 = []
        colors_and_percentage = []
        kmax = 20
        min_per = 0.0
        images_to_use = cropped_images[:ntest]
        color_res = color_labels[:ntest]

        results = benchmark_kmeans_initializations(
            images_to_use,
            color_res,
            clusterer_class=KMeans,
            kmax=10,
            iters=20,
            min_per=0.0,
            decay_threshold=0.1
        )
        print(results)

    if RUN_DECAY_ANALYSIS:
        ntest = 180
        colors2 = []
        colors_and_percentage = []
        kmax = 20
        min_per = 0.0
        images_to_use = cropped_images[:ntest]
        color_res = color_labels[:ntest]

        decay_threshold = []
        accuracies = []
        k = 3

        for i in range(1, 21):
            colors2 = []
            colors_and_percentage = []

            for image in images_to_use:
                kmeans = KMeans(
                    image,
                    k,
                    options={'fitting': 'Fisher', 'decay_threshold': i / 100, 'km_init': 'first'}
                )
                kmeans.find_bestK(kmax)
                kmeans.fit()
                dominant_colors = kmeans.get_dominant_colors(min_per)
                colors_and_percentage.append(dominant_colors)
                colors2.append(list(set([color for color, _ in dominant_colors])))

            if RUN_DECAY_VISUALIZATION:
                Retrieval_by_color(imgs[:ntest], colors_and_percentage, 'Pink, Black')

            accuracy = Get_color_accuracy(colors2, color_res)
            print("Percentatge d'etiquetes correctes per", i / 100, ":", accuracy)
            accuracies.append(accuracy)
            decay_threshold.append(i / 100)

        plt.figure(figsize=(10, 6))
        plt.plot(decay_threshold, accuracies, marker='o', linestyle='-', color='blue')
        plt.title("Percentatge d'etiquetes correctes per taxa de millora mínima")
        plt.xlabel("Improvement Rate")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.xticks(decay_threshold, rotation=45)
        plt.tight_layout()
        plt.show()