import torch
from pathlib import Path
import pickle
from collections import defaultdict
import json
import jstyleson
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
import os.path as osp
# use bfloat16 for the entire notebook
dtype = torch.float32 #torch.float16

torch.autocast(device_type="cuda", dtype=dtype).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# torch.multiprocessing.set_start_method("spawn")

import colorsys
import datetime
import os
import subprocess

import cv2
import gradio as gr
import imageio.v2 as iio
import imageio.v3 as iio_v3
import numpy as np

from loguru import logger as guru

from sam2.build_sam import build_sam2_video_predictor


def dump_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height



def annotate_frames(frames, fontsize=30):
    annotated_frames = []
    font = ImageFont.load_default(fontsize)
    # font = ImageFont.truetype("arialbd.ttf", size=12)

    for idx, frame in enumerate(frames):
        # Convert numpy array to PIL Image if necessary
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)

        # Create a drawing context
        draw = ImageDraw.Draw(frame)

        # Define text and box parameters
        text = f'Frame {idx}'
        text_color = 'black'
        box_color = 'white'
        box_border = 'black'
        box_padding = 5

        # Calculate text size and box size
        text_width, text_height = textsize(text, font=font)
        # text_width, text_height = draw.textsize(text, font=font)
        box_width = text_width + 2 * box_padding
        box_height = text_height + 2 * box_padding

        image_width, image_height = frame.size
        text_position = (image_width - box_width - box_padding, image_height - box_height - box_padding)
        box_position = (image_width - box_width - box_padding, image_height - box_height - box_padding,
                        image_width - box_padding, image_height - box_padding)

        # Draw white box with black border at top left
        draw.rectangle(box_position, outline=box_border, fill=box_color)

        # Draw frame ID text in black bold font
        draw.text(text_position, text, font=font, fill=text_color, stroke_width=0)

        # Append annotated frame to list
        annotated_frames.append(np.array(frame))  # Convert back to numpy array

    return annotated_frames


class PromptGUI(object):
    def __init__(self, checkpoint_dir, model_cfg):
        self.checkpoint_dir = checkpoint_dir
        self.model_cfg = model_cfg
        self.sam_model = None
        self.tracker = None

        # frame_id -> mask_id -> x,y,label
        self.selected_points = {}
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        self.cur_mask_idx = 0
        # can store multiple object masks
        # saves the masks and logits for each mask index
        self.point_masks = {}

        self.index_masks_all = []
        self.color_masks_all = []

        self.img_dir = ""
        self.img_paths = []
        self.init_sam_model()

    def init_sam_model(self):
        if self.sam_model is None:
            self.sam_model = build_sam2_video_predictor(self.model_cfg, self.checkpoint_dir)
            guru.info(f"loaded model checkpoint {self.checkpoint_dir}")


    def clear_all_points(self) -> tuple[None, None, str]:
        self.selected_points[self.frame_index][self.cur_mask_idx].clear()
        message = "Cleared points, select new points to update mask"
        if self.frame_index in self.point_masks:
            _, obj_ids, out_mask_logits = self.sam_model.clear_all_prompts_in_frame(self.inference_state, self.frame_index, self.cur_mask_idx, need_output=True)
            mask = {
                out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                for i, out_obj_id in enumerate(obj_ids)
            }
            index_mask = self.make_index_mask(mask)
            self.point_masks[self.frame_index] = index_mask
            image = self.image.copy()

            palette = get_hls_palette(index_mask.max() + 1)
            color_mask = palette[index_mask]
            out_u = compose_img_mask(image, color_mask)
            out = draw_points(out_u, self.selected_points.get(self.frame_index, {}).get(self.cur_mask_idx, []))
            return out, None, message
        else:
            return self.image, None, message

    def make_index_mask(self, masks):
        assert len(masks) > 0
        idcs = list(masks.keys())
        idx_mask = masks[idcs[0]].astype("uint8")
        for i in idcs:
            mask = masks[i]
            idx_mask[mask] = i + 1
        return idx_mask

    def _clear_image(self):
        """
        clears image and all masks/logits for that image
        """
        self.image = None
        self.cur_mask_idx = 0
        self.frame_index = 0

        self.point_masks = {}
        self.selected_points = {}

        self.index_masks_all = []
        self.color_masks_all = []

    def reset(self):
        self._clear_image()
        if hasattr(self, "inference_state"):
            self.sam_model.reset_state(self.inference_state)
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def set_img_dir(self, img_dir: str) -> int:
        self._clear_image()
        self.img_dir = img_dir
        self.img_paths = [
            f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]
        return len(self.img_paths)

    def set_input_image(self, i: int = 0) -> np.ndarray | None:
        guru.debug(f"Setting frame {i} / {len(self.img_paths)}")
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.frame_index = i
        image = iio.imread(self.img_paths[i])
        self.image = image
        if i in self.point_masks:
            index_mask = self.point_masks[i]
            palette = get_hls_palette(index_mask.max() + 1)
            color_mask = palette[index_mask]
            out_u = compose_img_mask(image.copy(), color_mask)
            out = draw_points(out_u, self.selected_points.get(self.frame_index, {}).get(self.cur_mask_idx, []))
            return out
        return image

    def get_sam_features(self) -> tuple[str, np.ndarray | None]:
        self.inference_state = self.sam_model.init_state(video_path=self.img_dir)
        self.sam_model.reset_state(self.inference_state)
        msg = (
            "SAM features extracted. "
            "Click points to update mask, and submit when ready to start tracking"
        )
        return msg

    def set_positive(self) -> str:
        self.cur_label_val = 1.0
        return "Selecting positive points. Submit the mask to start tracking"

    def set_negative(self) -> str:
        self.cur_label_val = 0.0
        return "Selecting negative points. Submit the mask to start tracking"

    def add_point(self, frame_idx, i, j):
        """
        get the index mask of the objects
        """
        self.selected_points.setdefault(self.frame_index, {}).setdefault(self.cur_mask_idx, []).append([j, i, self.cur_label_val])
        pt = self.selected_points[self.frame_index][self.cur_mask_idx]

        # masks, scores, logits if we want to update the mask
        masks = self.get_sam_mask(
            frame_idx, np.array(pt, dtype=np.float32)[:, :2], np.array(pt, dtype=np.int32)[:, 2]
        )
        mask = self.make_index_mask(masks)
        self.point_masks[frame_idx] = mask
        return mask


    def get_sam_mask(self, frame_idx, input_points, input_labels):
        """
        :param frame_idx int
        :param input_points (np array) (N, 2)
        :param input_labels (np array) (N,)
        return (H, W) mask, (H, W) logits
        """
        assert self.sam_model is not None


        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        with torch.autocast(device_type="cuda", dtype=dtype): # , dtype=torch.bfloat16
            _, out_obj_ids, out_mask_logits = self.sam_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=self.cur_mask_idx,
                points=input_points,
                labels=input_labels,
            )

        return  {
                out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }


    def run_tracker(self) -> tuple[str, str]:

        # read images and drop the alpha channel
        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]

        video_segments = {}  # video_segments contains the per-frame segmentation results
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        with torch.autocast(device_type="cuda", dtype=dtype): # , dtype=torch.bfloat16
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(self.inference_state, start_frame_idx=0):
                masks = {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                video_segments[out_frame_idx] = masks
            # index_masks_all.append(self.make_index_mask(masks))

        self.index_masks_all = [self.make_index_mask(v) for k, v in video_segments.items()]

        out_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all)
        out_vidpath = "tracked_colors.mp4"
        # write frame ID
        out_frames = annotate_frames(out_frames)
        iio.mimwrite(out_vidpath, out_frames)
        message = f"Wrote current tracked video to {out_vidpath}."
        instruct = "Save the masks to an output directory if it looks good!"
        return out_vidpath, f"{message} {instruct}"

    def save_masks_to_dir(self, object_names, output_dir: str, task_name) -> str:
        assert self.color_masks_all is not None
        os.makedirs(osp.join(output_dir, 'imask'), exist_ok=True)
        os.makedirs(osp.join(output_dir, 'cmask'), exist_ok=True)
        dump_pkl(self.selected_points, f"{output_dir}/{task_name}.pkl")
        dump_json(object_names, f"{output_dir}/names.json")

        for frame_idx, (img_path, clr_mask, id_mask) in enumerate(zip(self.img_paths, self.color_masks_all, self.index_masks_all)):
            name = os.path.basename(img_path)
            out_path = f"{output_dir}/cmask/{name}"
            if frame_idx in self.point_masks:
                id_mask = self.point_masks[frame_idx]
                # id_mask = np.concatenate([self.point_masks[frame_idx][None], id_mask[None]], axis=0)
                # id_mask = id_mask.max(0)
                palette = get_hls_palette(id_mask.max() + 1)
                clr_mask =  palette[id_mask.astype("int")]
            iio.imwrite(out_path, clr_mask)
            np_out_path = f"{output_dir}/imask/{osp.splitext(name)[0]}.png"
            Image.fromarray(id_mask).save(np_out_path)
            # np.save(np_out_path, id_mask)

        message = f"Saved masks to {output_dir}!"
        guru.debug(message)
        gr.Info(message)
        return message


def isimage(p):
    ext = os.path.splitext(p.lower())[-1]
    return ext in [".png", ".jpg", ".jpeg"]


def draw_points(img, points):
    out = img.copy()
    for p in points:
        x, y, label = int(p[0]), int(p[1]), int(p[2])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 3, color, -1)
    return out


def get_hls_palette(
    n_colors: int,
    lightness: float = 0.5,
    saturation: float = 0.7,
) -> np.ndarray:
    """
    returns (n_colors, 3) tensor of colors,
        first is black and the rest are evenly spaced in HLS space
    """
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1]  # (n_colors - 1)
    # hues = (hues + first_hue) % 1
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")


def colorize_masks(images, index_masks, fac: float = 0.4):
    max_idx = max([m.max() for m in index_masks])
    guru.debug(f"{max_idx=}")
    palette = get_hls_palette(max_idx + 1)
    color_masks = []
    out_frames = []
    for img, mask in zip(images, index_masks):
        clr_mask = palette[mask.astype("int")]
        color_masks.append(clr_mask)
        out_u = compose_img_mask(img, clr_mask, fac)
        out_frames.append(out_u)
    return out_frames, color_masks


def compose_img_mask(img, color_mask, fac: float = 0.6, dont_dim_background=True):
    if dont_dim_background:
        color_mask = color_mask.copy()
        bg_mask = color_mask.sum(2) == 0
        color_mask[bg_mask] = img[bg_mask]
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    out_u = (255 * out_f).astype("uint8")
    return out_u


def listdir(vid_dir):
    if vid_dir is not None and os.path.isdir(vid_dir):
        return sorted(os.listdir(vid_dir))
    return []


def make_demo(
    checkpoint_dir,
    model_cfg,
    tasks,
):
    tasks = {t['name']: t for t in tasks}

    def select_media(task_name):
        guru.debug(f"Selected task: {task_name}")
        import uuid
        tmp_path = Path(f"/tmp/{uuid.uuid4()}")
        tmp_vid_path = str(tmp_path / "tmp.mp4")
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = tasks[task_name]["frames"]
        frames_lst = [iio.imread(p) for p in frame_paths]
        frames = np.array(frames_lst)
        anno_frames = annotate_frames(frames)
        iio.mimwrite(tmp_vid_path, anno_frames)

        prompts.reset()
        img_dir.mkdir(parents=True, exist_ok=True)
        for fp, frame in zip(frame_paths, frames_lst):
            bn = osp.basename(fp).split(".")[0]
            iio.imwrite(img_dir / f"{bn}.jpg", Image.fromarray(frame).convert("RGB"))
        prompts.set_img_dir(str(img_dir))
        return prompts.set_input_image(0), gr.update(minimum=0, maximum=len(prompts.img_paths) - 1, value=0), tmp_vid_path, "Now `Click to Load SAM and Prev Annotations!` to start annotating.", "Click to Load SAM and Prev Annotations!"

    def get_select_coords(frame_idx, img, evt: gr.SelectData):
        i = evt.index[1]  # type: ignore
        j = evt.index[0]  # type: ignore
        try:
            index_mask = prompts.add_point(frame_idx, i, j)
        except AttributeError:
            gr.Warning("Please `Click to Load SAM and Prev Annotations!` first")
            return None
        guru.debug(f"{index_mask.shape=}")
        palette = get_hls_palette(index_mask.max() + 1)
        color_mask = palette[index_mask]
        out_u = compose_img_mask(prompts.image, color_mask)
        out = draw_points(out_u, prompts.selected_points.get(prompts.frame_index, {}).get(prompts.cur_mask_idx, []))
        return out

    prompts = PromptGUI(checkpoint_dir, model_cfg)

    start_instructions = (
        "Select a media to annotate. The objects shall separated by comma."
    )
    with gr.Blocks(css="#pos_button {background-color: green}  .pos { border: 8px solid green} .neg {border: 8px solid red}") as demo:
        instruction = gr.Textbox(
            start_instructions, label="Instruction", interactive=False
        )

        with gr.Row():
            with gr.Column():
                sam_button = gr.Button("Click to Load SAM and Prev Annotations!")
                # reset_button = gr.Button("Reset (click this when start annotating a new video)")
                # def reset_button_click():
                #     prompts.reset()
                #     return "Click to Load SAM and Prev Annotations!"
                # reset_button.click(reset_button_click, outputs=[sam_button])

                media_dropdown = gr.Dropdown(
                    label="Medias", choices=list(tasks.keys()), value=None
                )

                frame_index = gr.Slider(
                    label="Frame index",
                    minimum=0,
                    maximum=len(prompts.img_paths) - 1,
                    value=0,
                    step=1,
                )

                with gr.Row():
                    pos_button = gr.Button("Toggle positive", variant='primary', elem_id='pos_button')
                    neg_button = gr.Button("Toggle negative", variant='stop')
                clear_button = gr.Button("Clear current Object at this Frame")

                with gr.Accordion("Watch Origin Video"):
                    origin_video = gr.Video()

            with gr.Column():
                input_image = gr.Image(
                    prompts.set_input_image(0),
                    label="Image",
                    every=1,
                    elem_classes='pos'
                )

            with gr.Column():
                def str2opts(s):
                    return [a.strip() for a in s.strip().split(',')]

                object_names = gr.Textbox(
                    "object", label="Objects", interactive=True
                )
                mask_dropdown = gr.Dropdown(label="Curr Object", choices=str2opts(object_names.value), interactive=True)
                object_names.change(lambda onames: gr.update(choices=str2opts(onames), value=str2opts(onames)[0]), [object_names], [mask_dropdown])

                def select_mask(object_names, select_name):
                    objects = str2opts(object_names) # mask index is the order of the object names
                    index = objects.index(select_name)
                    prompts.cur_mask_idx = index
                    # shall update point visualization
                    prompts.set_positive()
                    img = prompts.set_input_image(prompts.frame_index)
                    return gr.update(value=img, elem_classes='pos'), f"Annotating {select_name}"

                mask_dropdown.select(select_mask, [object_names, mask_dropdown], [input_image, instruction])
                submit_button = gr.Button("🐷 Start Mask Tracking ")
                final_video = gr.Video(label="Masked video")
                save_button = gr.Button("Save masks (merge tracked and annotated)")

        media_dropdown.select(select_media, [media_dropdown], outputs=[input_image, frame_index, origin_video, instruction, sam_button])
        # select_media(media_paths[0])

        frame_index.change(prompts.set_input_image, [frame_index], [input_image, ])
        input_image.select(get_select_coords, [frame_index, input_image], [input_image])
        clear_button.click(prompts.clear_all_points, outputs=[input_image, final_video, instruction])

        def sam_button_click(task_name, onames):
            txt = [gr.update(value="SAM and Annotations Loaded!"), prompts.image, onames, prompts.get_sam_features()]
            media_path = tasks[task_name]["output_path"]
            anno_path = Path(media_path)

            txt[2] = ",".join(tasks[task_name]["instances"])
            if (anno_path / "names.json").exists():
                object_names = json.load((anno_path / "names.json").open())
                txt[2] = ",".join(object_names)

            if (anno_path / f"{task_name}.pkl").exists():
                gr.Info("Loading previous annotations")
                prompts.selected_points = pickle.loads(Path(anno_path / f"{task_name}.pkl").read_bytes())
                for frame_idx, points in prompts.selected_points.items():
                    for mask_idx, pts in points.items():
                        prompts.cur_mask_idx = mask_idx
                        masks = prompts.get_sam_mask(
                            frame_idx, np.array(pts, dtype=np.float32)[:, :2], np.array(pts, dtype=np.int32)[:, 2]
                        )
                        mask = prompts.make_index_mask(masks)
                        prompts.point_masks[frame_idx] = mask
                txt[1] = prompts.set_input_image(prompts.frame_index)

            return txt

        sam_button.click(sam_button_click, [media_dropdown, object_names], outputs=[sam_button, input_image, object_names, instruction])

        def set_positive():
            return gr.update(elem_classes='pos'), prompts.set_positive()

        def set_negative():
            return gr.update(elem_classes='neg'), prompts.set_negative()

        pos_button.click(set_positive, outputs=[input_image, instruction])
        neg_button.click(set_negative, outputs=[input_image, instruction])

        submit_button.click(prompts.run_tracker, outputs=[final_video, instruction])
        save_button.click(
            lambda onames, task_name: prompts.save_masks_to_dir(str2opts(onames),
                                                                 tasks[task_name]["output_path"], task_name),
            [object_names, media_dropdown],
            outputs=[instruction]
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml", help="path of the model config file, relative to the *.pt model file")
    parser.add_argument("--task_json", type=str, required=True)
    args = parser.parse_args()
    tasks = jstyleson.load(open(args.task_json))
    if isinstance(tasks, dict):
        tasks = [tasks] 
    for task in tasks:
        if "root" in task:
            if not task['root']:
                task['root'] = osp.abspath(osp.dirname(args.task_json))
            task['frames'] = [osp.join(task['root'], f) for f in task['frames']]
            task['output_path'] = osp.join(task['root'], task['output_path'])

    demo = make_demo(
        args.checkpoint,
        args.model_cfg,
        tasks
    )
    demo.launch(server_port=args.port, share=os.environ.get('share', 'false').lower() == 'true')
