from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

from arc2face import CLIPTextModelWrapper, project_face_embs

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import os
import sys
sys.path.append('..\\FaceCoresetNet')
from train_val_template import FaceCoresetNet
from utils import dotdict
import config

args = config.get_args()
hparams = dotdict(vars(args))
base_model = 'runwayml/stable-diffusion-v1-5'

facecoreset_model = FaceCoresetNet(**hparams)

model_weights = torch.load('D:\gil\deep_generative_models\FaceCoresetNet\experiments\\arcface_arc2face_RGB_enter_model_glr=0.001_lr=0.001_05-18_1\epoch=23-step=65544.ckpt')
facecoreset_model.load_state_dict(model_weights['state_dict'], strict=True )

facecoreset_model.eval()
facecoreset_model.to('cuda:0')
# with torch.no_grad():
#     facecoreset_model.aggregate_model.gamma.fill_(0.6)

encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="encoder", torch_dtype=torch.float16
)

unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=torch.float16
)

pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None
    )


pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to('cuda')


app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#app = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)

#img = np.array(Image.open('assets/examples/joacquin.png'))[:,:,::-1]
template_dir = 'assets/template/condolisa_8'


files = os.listdir(template_dir)
template_embs = []
#mask = [False, False, False, False, False, True, True, True]

#i = 0
for file in files:
    full_path = os.path.join(template_dir, file)
    img = np.array(Image.open(full_path))[:, :, ::-1]
    faces = app.get(img)
    try:
        faces = sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
            -1]  # select largest face (if more than one detected)
    except:
        print('failed face detection on ' + full_path )
        continue
    #id_emb = torch.tensor(faces['embedding'], dtype=torch.float16)[None].cuda()
    id_emb = torch.tensor(faces['embedding'], dtype=torch.float32)[None].cuda()
    #if mask[i]:
    template_embs.append(id_emb)
    #i += 1

template_embs = torch.cat(template_embs)

template_embs_avg = template_embs.sum(dim=0)
template_embs_avg = template_embs_avg / template_embs.norm()
template_embs_avg = template_embs_avg.unsqueeze(0)

norms = template_embs.norm(dim=1)
n_embs = torch.nn.functional.normalize(template_embs, p=2.0, dim=-1)
norms = norms.unsqueeze(dim=0).unsqueeze(dim=-1)
n_embs = n_embs.unsqueeze(0)
aggregate_embeddings, aggregate_norms, FPS_sample = facecoreset_model(embeddings=n_embs, norms=norms, only_FPS=True)

FPS_sample = FPS_sample.squeeze(0)
FPS_sample = FPS_sample.sum(dim=0)
FPS_sample = torch.nn.functional.normalize(FPS_sample, p=2.0, dim=-1)
FPS_sample = FPS_sample.unsqueeze(0)
#aggregate_embeddings = FPS_sample
#aggregate_embeddings = FPS_sample
#agg_type = 'facecoresetnet'
agg_type = 'avg'
if agg_type == 'avg':
    template_embs = template_embs_avg
else:
    template_embs = aggregate_embeddings
#template_emb = template_embs/torch.norm(template_embs, dim=1, keepdim=True)   # normalize embedding
id_emb = torch.tensor(template_embs, dtype=torch.float16)[None].cuda()
id_emb = project_face_embs(pipeline, id_emb)
num_images = 4
images = pipeline(prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images

for i, image in enumerate(images):
    image.save(os.path.join('./output', str(i)+'.png'))

# faces = app.get(img)
# faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
# id_emb = torch.tensor(faces['embedding'], dtype=torch.float16)[None].cuda()
# id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
# id_emb = project_face_embs(pipeline, id_emb)
# num_images = 4
# images = pipeline(prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images
pass