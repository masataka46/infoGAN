import numpy as np
# import os
from PIL import Image
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import csv


def compute_precision_recall(score_A_np):
    array_5 = np.where(score_A_np[:, 1] == 5.0)
    array_7 = np.where(score_A_np[:, 1] == 7.0)
    print("len(array_5), ", len(array_5))
    print("len(array_7), ", len(array_7))

    mean_5 = np.mean((score_A_np[array_5])[:, 0])
    mean_7 = np.mean((score_A_np[array_7])[:, 0])
    medium = (mean_5 + mean_7) / 2.0
    print("mean_5, ", mean_5)
    print("mean_7, ", mean_7)
    print("medium, ", medium)

    array_upper = score_A_np[:, 0] >= medium
    array_lower = score_A_np[:, 0] < medium
    print("np.sum(array_upper.astype(np.float32)), ", np.sum(array_upper.astype(np.float32)))
    print("np.sum(array_lower.astype(np.float32)), ", np.sum(array_lower.astype(np.float32)))
    array_5_tf = score_A_np[:, 1] == 5.0
    array_7_tf = score_A_np[:, 1] == 7.0
    print("np.sum(array_5_tf.astype(np.float32)), ", np.sum(array_5_tf.astype(np.float32)))
    print("np.sum(array_7_tf.astype(np.float32)), ", np.sum(array_7_tf.astype(np.float32)))

    tn = np.sum(np.equal(array_lower, array_5_tf).astype(np.int32))
    tp = np.sum(np.equal(array_upper, array_7_tf).astype(np.int32))
    fp = np.sum(np.equal(array_upper, array_5_tf).astype(np.int32))
    fn = np.sum(np.equal(array_lower, array_7_tf).astype(np.int32))

    precision = tp / (tp + fp + 0.00001)
    recall = tp / (tp + fn + 0.00001)

    return tp, fp, tn, fn, precision, recall


def save_graph(x, y, filename, epoch):
    plt.plot(x, y)
    plt.title('ROC curve ' + filename + ' epoch:' + str(epoch))
    # x axis label
    plt.xlabel("FP / (FP + TN)")
    # y axis label
    plt.ylabel("TP / (TP + FN)")
    # save
    plt.savefig(filename + '_ROC_curve_epoch' + str(epoch) +'.png')
    plt.close()


def make_ROC_graph(score_A_np, filename, epoch):
    argsort = np.argsort(score_A_np, axis=0)[:, 0]
    score_A_np_sort = score_A_np[argsort][::-1]
    value_1_0 = (np.where(score_A_np_sort[:, 1] == 7., 1., 0.)).astype(np.float32)
    # score_A_np_sort_0_1 = np.concatenate((score_A_np_sort, value_1_0), axis=1)
    sum_1 = np.sum(value_1_0)

    len_s = len(score_A_np)
    sum_0 = len_s - sum_1
    tp = np.cumsum(value_1_0).astype(np.float32)
    index = np.arange(1, len_s + 1, 1).astype(np.float32)
    fp = index - tp
    fn = sum_1 - tp
    tn = sum_0 - fp
    tp_ratio = tp / (tp + fn + 0.00001)
    fp_ratio = fp / (fp + tn + 0.00001)
    save_graph(fp_ratio, tp_ratio, filename, epoch)

    auc = sm.auc(fp_ratio, tp_ratio)
    return auc


def unnorm_img(img_np):
    img_np_255 = (img_np + 1.0) * 127.5
    img_np_255_mod1 = np.maximum(img_np_255, 0)
    img_np_255_mod1 = np.minimum(img_np_255_mod1, 255)
    img_np_uint8 = img_np_255_mod1.astype(np.uint8)
    return img_np_uint8


def convert_np2pil(images_255):
    list_images_PIL = []
    for num, images_255_1 in enumerate(images_255):
        # img_255_tile = np.tile(images_255_1, (1, 1, 3))
        image_1_PIL = Image.fromarray(images_255_1)
        list_images_PIL.append(image_1_PIL)
    return list_images_PIL
    
def make_output_img(img_batch_list, epoch, log_file_name, out_img_dir):
    (data_num, img1_h, img1_w, _) = img_batch_list[0].shape
    wide_image_np = np.ones(((img1_h + 1) * data_num - 1, (img1_w + 1) * len(img_batch_list) -1, 3), dtype=np.uint8) * 255
    wide_image_PIL = Image.fromarray(wide_image_np)
    for num_b, img_batch in enumerate(img_batch_list):
        img_batch_unn = np.tile(unnorm_img(img_batch), (1, 1, 3))
        img_batch_PIL = convert_np2pil(img_batch_unn)
        for num, img_1 in enumerate(img_batch_PIL):
            wide_image_PIL.paste(img_1, (num_b * (img1_w + 1), num * (img1_h + 1)))

    wide_image_PIL.save(out_img_dir + "/resultImage_"+ log_file_name + '_' + str(epoch) + ".png")





def save_list_to_csv(list, filename):
    f = open(filename, 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(list)
    f.close()






