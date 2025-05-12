import pickle as pkl
from PIL import Image
from pathlib import Path
import torch
import numpy as np  
from PIL import Image, ImageDraw
from typing import *
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    keypoint_scores: Optional[torch.Tensor] = None,
    keypoint_names: Optional[List[str]] = None,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = (255, 0, 0),
    line_color = 'white',
    radius: int = 2,
    width: int = 3,
    output_pil=True,
    transparency=1.0,
    line_under=True
) -> torch.Tensor:
    def is_valid(*args):
        return all([a >= 0 for a in args])
    
    if isinstance(image, Image.Image):
        image = pil_to_tensor(image)
    
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
    
    POINT_SIZE = keypoints.shape[-1]
    if isinstance(keypoints, np.ndarray):
        keypoints = torch.from_numpy(keypoints)
    
    keypoints = keypoints.reshape(1, -1, POINT_SIZE)

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    if transparency < 1.0:
        draw = ImageDraw.Draw(img_to_draw, 'RGBA')
    else:
        draw = ImageDraw.Draw(img_to_draw, None if POINT_SIZE == 2 else 'RGBA')
    keypoints = keypoints.clone()
    if POINT_SIZE == 3:
        keypoints[:, :, -1] *= 255
    img_kpts = keypoints.to(torch.int64).tolist()

    for kpt_id, kpt_inst in enumerate(img_kpts):
        kpt_size = len(kpt_inst[0])

        def draw_line():
            if connectivity is not None:
                for connection in connectivity:
                    start_pt_x = kpt_inst[connection[0]][0]
                    start_pt_y = kpt_inst[connection[0]][1]

                    end_pt_x = kpt_inst[connection[1]][0]
                    end_pt_y = kpt_inst[connection[1]][1]

                    if not is_valid(start_pt_x, start_pt_y, end_pt_x, end_pt_y):
                        continue

                    if transparency < 1.0:
                        kp_line_color = line_color + (int(255*(1- transparency)), )
                    else:
                        kp_line_color = line_color

                    draw.line(
                        ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                        width=width, fill=kp_line_color
                    )
        
        def draw_points():
            for inst_id, kpt in enumerate(kpt_inst):
                if not is_valid(*kpt):
                    continue
                x1 = kpt[0] - radius
                x2 = kpt[0] + radius
                y1 = kpt[1] - radius
                y2 = kpt[1] + radius
                if len(kpt) == 3:
                    kp_color = colors + (int(kpt[2]), )
                elif transparency < 1.0:
                    kp_color = colors + (int(255*(1- transparency)), )
                else:
                    kp_color = colors
                draw.ellipse([x1, y1, x2, y2], fill=kp_color, outline=None, width=0)
                txt = ''
                if keypoint_scores is not None:
                    txt += f'{float(keypoint_scores[inst_id]):.2f}'
                if keypoint_names is not None:
                    if keypoint_scores is not None:
                        txt += ','
                    txt += f'{keypoint_names[inst_id]}'
                if txt:
                    x1, y1 = x1 + 5, y1 + 5
                    draw.rectangle([x1, y1, x1 + 5*len(txt), y1 + 10], fill=(255, 255, 255))
                    draw.text((x1, y1), txt, fill=(0,0,0), font_size=10)
        
        if line_under:
            draw_line()
            draw_points()
        else:
            draw_points()
            draw_line()
            
    if output_pil:
        return img_to_draw  
    else:
        return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)

