
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


import pandas as pd
import re
import shutil
# import net
import torch
import os
import torchvision
from face_alignment import align
import numpy as np
# import PIL
from PIL import ImageFilter
from PIL import Image
from utils import dotdict
import config
import train_val_template as train_val
from itertools import combinations
from sklearn.metrics import roc_curve, auc


def set_postive_pairs(pairs, templates):

    tt = {}
    for i, t in enumerate(templates):
        tt[t['id']] = i

    # pairs = {(tt['VID-20231008-WA0033.faces_1'], tt['VID-20231008-WA0033.faces_3']): True}
    # return pairs
    pairs[(tt['VID-20231008-WA0033.faces_1'], tt['VID-20231008-WA0033.faces_3'])] = 1
    pairs[(tt['VID-20231008-WA0035.faces_3'], tt['VID-20231008-WA0035.faces_4'])] = 1
    pairs[(tt['VID-20231008-WA0035.faces_5'], tt['VID-20231008-WA0035.faces_10'])] = 1
    pairs[(tt['VID-20231008-WA0035.faces_16'], tt['VID-20231008-WA0035.faces_19'])] = 1
    pairs[(tt['VID-20231008-WA0035.faces_21'], tt['VID-20231008-WA0035.faces_22'])] =  1
    pairs[(tt['VID-20231008-WA0035.faces_23'], tt['VID-20231008-WA0035.faces_24'])] = 1
    pairs[(tt['VID-20231008-WA0035.faces_21'], tt['VID-20231008-WA0035.faces_36'])] = 1
    pairs[(tt['VID-20231008-WA0035.faces_38'], tt['VID-20231008-WA0035.faces_39'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_6'], tt['VID-20231008-WA0039.faces_7'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_6'], tt['VID-20231008-WA0039.faces_25'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_6'], tt['VID-20231008-WA0039.faces_31'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_6'], tt['VID-20231008-WA0039.faces_20'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_7'], tt['VID-20231008-WA0039.faces_25'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_7'], tt['VID-20231008-WA0039.faces_31'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_7'], tt['VID-20231008-WA0039.faces_20'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_20'], tt['VID-20231008-WA0039.faces_25'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_25'], tt['VID-20231008-WA0039.faces_31'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_20'], tt['VID-20231008-WA0039.faces_31'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_6'], tt['VID-20231008-WA0039.faces_55'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_7'], tt['VID-20231008-WA0039.faces_55'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_25'], tt['VID-20231008-WA0039.faces_55'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_31'], tt['VID-20231008-WA0039.faces_55'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_20'], tt['VID-20231008-WA0039.faces_55'])] =1
    pairs[(tt['VID-20231008-WA0039.faces_3'], tt['VID-20231008-WA0039.faces_55'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_3'], tt['VID-20231008-WA0039.faces_6'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_3'], tt['VID-20231008-WA0039.faces_7'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_3'], tt['VID-20231008-WA0039.faces_25'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_3'], tt['VID-20231008-WA0039.faces_31'])] = 1
    pairs[(tt['VID-20231008-WA0039.faces_3'], tt['VID-20231008-WA0039.faces_20'])] = 1
    pairs[(tt['VID-20231008-WA0041.faces_6'], tt['VID-20231008-WA0041.faces_10'])] = 1
    pairs[(tt['VID-20231008-WA0041.faces_8'], tt['VID-20231008-WA0041.faces_15'])] = 1

    return pairs


def compute_pairs_labels(templates ):
    #template_names = [t['id'] for t in templates]
    template_indexes = range(len(templates))
    t_pairs = list(combinations(template_indexes, 2))
    pairs = {}
    for p in t_pairs:
        pairs[p] = 0

    pairs = set_postive_pairs(pairs, templates)
    return pairs


def compute_score_for_pairs(template,similarity_scores):
    score_for_pairs = {}
    template_names = [t['id'] for t in templates]
    for i in range(0, len(template_names)):
        for j in range(i + 1, len(template_names)):
            score_for_pairs[(template_names[i],template_names[j])] = similarity_scores[i,j]

    return score_for_pairs

def create_roc(templates, similarity_scores):

    score_for_pairs = compute_score_for_pairs(templates, similarity_scores)
    pairs_labels = compute_pairs_labels(templates )

    pairs_labels_list = list(pairs_labels.items())
    pairs_labels_list = [i[1] for i in pairs_labels_list]

    score_list = list(score_for_pairs.items())
    score_list = [i[1].detach().cpu().numpy() for i in score_list]

    fpr, tpr, _ = roc_curve(pairs_labels_list,score_list)
    np.savetxt('fpr.txt', fpr)
    np.savetxt('tpr.txt', tpr)
    # roc_auc = auc(fpr, tpr)
    # fpr = np.flipud(fpr)
    # tpr = np.flipud(tpr)  # select largest tpr at same fpr
    # plt.plot(fpr,
    #          tpr,
    #          color=colours[method],
    #          lw=1,
    #          label=('[%s (AUC = %0.4f %%)]' %
    #                 (method.split('-')[-1], roc_auc * 100)))

    pass






# adaface_models = {\
#     'ir_101':"pretrained/adaface_ir101_ms1mv2.ckpt",
#     'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
#     #:"pretrained/adaface_ir18_vgg2.ckpt",
#     'ir_18':'experiments/run_ir18_ms1mv2_subset_04-22_5/epoch=24-step=45650.ckpt'}

# def load_pretrained_model(architecture='ir_50'):
#     # load model and pretrained statedict
#     assert architecture in adaface_models.keys()
#     model = net.build_model(architecture)
#     statedict = torch.load(adaface_models[architecture])['state_dict']
#     model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
#     model.load_state_dict(model_statedict)
#     model.eval()
#     return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    #bgr_img_hwc = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    bgr_img_hwc = ((np_img / 255.) - 0.5) / 0.5
    # rgb_img = ((np_img / 255.) - 0.5) / 0.5
    # rgb = > rbg
    tensor_chw = torch.tensor([bgr_img_hwc.transpose(2, 0, 1)]).float()
    # tensor = torch.tensor([rgb_img.transpose(1, 2, 0)]).float()
    return tensor_chw


#def build_dataframe(template_root):



# Function to extract face ID and frame ID from filename
def extract_ids(filename):
    match = re.match(r'face_(\d+)_(\d+)\.png', filename)
    if match:
        face_id = int(match.group(1))
        frame_id = int(match.group(2))
        return face_id, frame_id
    else:
        return None, None


# Function to process video frames in a directory
def process_video_directory(video_dir, dataframe):
    video_name = os.path.basename(video_dir)
    for filename in os.listdir(video_dir):
        face_id, frame_id = extract_ids(filename)
        if face_id is not None and frame_id is not None:
            dataframe = dataframe.append({'VideoName': video_name, 'FaceID': face_id, 'FrameID': frame_id, 'FileName': filename},
                                         ignore_index=True)
    return dataframe


# Function to process all video directories in the input directory
def process_input_directory(input_dir):
    dataframe = pd.DataFrame(columns=['VideoName', 'FaceID', 'FrameID'])
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for dirname in dirnames:
            video_dir = os.path.join(dirpath, dirname)
            dataframe = process_video_directory(video_dir, dataframe)
    return dataframe



# Define the source directory containing the images
source_directory = 'images'


def create_qeuries_directory(source_directory, target_directory):

    # Iterate through the image files in the source directory
    for filename in os.listdir(source_directory):
        file_extension = os.path.splitext(filename)[1].lower()

        # Check if the file is an image (you can add more image extensions as needed)
        if file_extension in ('.jpg', '.jpeg', '.png', '.bmp', '.gif'):
        #if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):

            # Extract 'first_name' and 'last_name' from the filename using regular expressions
            match = re.match(r'^[A-Za-z]+_[A-Za-z]+', filename)
            print(filename)
            assert match

            name = match.group(0)
            # Create a directory based on 'first_name' and 'last_name'
            target_dir = os.path.join(target_directory, f'{name}')

            # Ensure the directory exists or create it if it doesn't
            os.makedirs(target_dir, exist_ok=True)

            # Copy the image to the proper directory
            source_path = os.path.join(source_directory, filename)
            target_path = os.path.join(target_dir, filename)
            print('writing ' + target_path)
            shutil.copy(source_path, target_path)
            print(f'Copied {filename} to {target_directory}')




if __name__ == '__main__':


    input_dir = 'D:/faces/all_images-20231013T200525Z-001/all_images/missing faces/'
    output_dir = 'D:/faces/queries'

    create_qeuries_directory(input_dir, output_dir)



    exit(0)


    args = config.get_args()
    hparams = dotdict(vars(args))
    center_crop = torchvision.transforms.CenterCrop(112)
    resize = torchvision.transforms.Resize(130)
    model = train_val.FaceCoresetNet(**hparams)
    checkpoint = torch.load(args.resume_from_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    #model.aggregate_model.gamma = torch.nn.Parameter(torch.tensor(0.0))
    model.eval()

    #template_root = 'D:/exp/FaceCoresetNet/qualitative_exp/'
    #template_root = 'D:/exp/FaceCoresetNet/sample/templates/'
    template_root = 'D:\exp\FaceCoresetNet\sample\sample_orig'
    #template_root = 'D:\exp\FaceCoresetNet\sample\small_sample'
    #template_root = 'D:\exp\FaceCoresetNet\sample\\test'

    dataframe = process_input_directory(template_root)

    grouped = dataframe.groupby(['VideoName', 'FaceID'])
    templates = []

    for key, group in grouped:
        video_name, face_id = key
        id_in_group = range(len(group['FrameID']))
        template = {}
        template['id'] = '_'.join((video_name, str(face_id)))
        template['images'] = []
        template['nearest_neighbors'] = []

        for id in id_in_group:

            row = group.iloc[id]
            path = os.path.join(template_root, row['VideoName'], row['FileName'])
            print('reading ' + path)
            img = Image.open(path).convert('RGB')
            img = resize(img)
            img = center_crop(img)
            aligned_rgb_img = img

            # aligned_rgb_img = align.get_aligned_face(img)
            # if aligned_rgb_img == None:
            #     rot_90 = img.rotate(90)
            #     aligned_rgb_img = align.get_aligned_face(rot_90)
            #     if aligned_rgb_img == None:
            #         rot_270 = img.rotate(270)
            #         aligned_rgb_img = align.get_aligned_face(rot_270)
            #         if aligned_rgb_img == None:
            #             continue

            input = to_input(aligned_rgb_img).unsqueeze(1)
            template['images'].append(input)

        template_tensor = torch.cat(template['images'], dim=1)

        aggregate_embeddings, aggregate_norms, _, _ = model(templates=template_tensor, labels=None, embeddings=None, norms=None,
                                              compute_feature=True, only_FPS=True)
        template['embedding'] = aggregate_embeddings
        template['embedding_norm'] = aggregate_norms
        templates.append(template)

    templates_tensor = [t['embedding'] for t in templates]
    templates_tensor = torch.cat(templates_tensor, dim=0)
    similarity_scores = templates_tensor @ templates_tensor.T

    create_roc(templates, similarity_scores)




    queries = templates[0]['embedding']
    similarity_scores = queries @ templates_tensor.T
    k = 3
    topk_values, topk_indices = torch.topk(similarity_scores, k=k, dim=1)
    for template_index in range(len(templates)):
        templates[template_index]['NN'] = \
            [templates[topk_indices[template_index,i]]['id'] + ': ' + str(topk_values[template_index][i].detach().numpy()) for i in range(k)]

    pass




    # templates_norms = []
    # templates_embedding = []
    # template = []
    # template_dir_list = sorted(os.listdir(template_root))
    # for template_dir in template_dir_list:
    #     for file in os.listdir(os.path.join(template_root, template_dir)):
    #         path = os.path.join(template_root, template_dir, file)
    #         print('reading ' + path)
    #         img = Image.open(path).convert('RGB')
    #
    #         aligned_rgb_img = align.get_aligned_face(img)
    #         if aligned_rgb_img == None:
    #             rot_90 = img.rotate(90)
    #             aligned_rgb_img = align.get_aligned_face(rot_90)
    #             if aligned_rgb_img == None:
    #                 rot_270 = img.rotate(270)
    #                 aligned_rgb_img = align.get_aligned_face(rot_270)
    #                 if aligned_rgb_img == None:
    #                     continue
    #
    #         input = to_input(aligned_rgb_img).unsqueeze(1)
    #         template.append(input)
    #
    #     template_tensor = torch.cat(template, dim=1)
    #
    #     aggregate_embeddings, aggregate_norms, _, _ = model(templates=template_tensor, labels=None, embeddings=None, norms=None,
    #                                           compute_feature=True, only_FPS=False)
    #     templates_embedding.append(aggregate_embeddings)
    #     templates_norms.append(aggregate_norms)
    #
    # templates_tensor = torch.cat(templates_embedding, dim=1)
    #
    # similarity_scores = templates_tensor @ templates_tensor.T
