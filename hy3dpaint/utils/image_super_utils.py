# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import numpy as np
import torch
from PIL import Image


class imageSuperNet:
    def __init__(self, config) -> None:
        import spandrel

        model = spandrel.ModelLoader().load_from_file(config.realesrgan_ckpt_path)
        model = model.eval().cuda()
        self.model = model

    @torch.no_grad()
    def __call__(self, image):
        img = np.array(image).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.cuda.amp.autocast(enabled=True):
            output = self.model(img)
        output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
        output = (output * 255).astype(np.uint8)
        return Image.fromarray(output)
