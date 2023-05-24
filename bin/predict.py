import os
import sys
import yaml
import uuid
import cv2
import torch
import logging
import traceback
import numpy as np
from datetime import datetime
import PIL.Image as Image

from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
from saicinpainting.evaluation.utils import move_to_device

# Initialize logger
LOGGER = logging.getLogger(__name__)

# Load prediction configuration from YAML file
with open('./../configs/prediction/default.yaml', 'r') as file:
    predict_config = OmegaConf.create(yaml.safe_load(file))

# load_model function
def load_model():
    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(predict_config.model.path,
                                       'models',
                                       predict_config.model.checkpoint)

        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cuda')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        return dict(model=model, device=device, config=predict_config)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)
    return None


# Load image function
def load_image(file, mode='RGB', return_orig=False):
    img = np.array(Image.open(file).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img

# Inpainting image function
def inpainting_image(params, images):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print('receive to process ', current_time)
    try:
        model = params['model']
        device = params['device']
        predict_config: OmegaConf = params['config']

        image = load_image(images['image'], mode='RGB')
        mask = load_image(images['mask'], mode='L')

        result = dict(image=image, mask=mask[None, ...])
        filename = str(uuid.uuid4());
        cur_out_fname = predict_config.outdir + '/' + filename + '.png'

        os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

        batch = default_collate([result])

        with torch.no_grad():
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('start ', current_time)
            batch = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1
            
            batch = model(batch)
            cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('End ', current_time)
            unpad_to_size = batch.get('unpad_to_size', None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(cur_out_fname, cur_res)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('Complete image draw ', current_time)

        return filename;

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)
    return None


if __name__ == '__main__':
    result = load_model()
    inpainting_image(result, True);
