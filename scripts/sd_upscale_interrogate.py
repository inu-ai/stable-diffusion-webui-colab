import math

import modules.scripts as scripts
import gradio as gr
from PIL import Image

from modules import processing, shared, sd_samplers, images, devices
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

if cmd_opts.deepdanbooru:
    from modules.deepbooru import get_deepbooru_tags

class Script(scripts.Script):
    def title(self):
        return "SD upscale interrogate"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        info = gr.HTML("<p style=\"margin-bottom:0.75em\">Will upscale the image to twice the dimensions; use width and height sliders to set tile size; fix the seed; will interrogate the prompt from the tiles; set the position of the interrogated prompt</p>")
        overlap = gr.Slider(minimum=0, maximum=256, step=16, label='Tile overlap', value=64, visible=False)
        upscaler_index = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index", visible=False)
        
        fix_seed = gr.Checkbox(label='Fix seed', value=True, visible=False)
        
        interrogate_type_choices = ['None', 'CLIP']
        if cmd_opts.deepdanbooru:
            interrogate_type_choices.append('DeepBooru')
        interrogate_type = gr.Radio(label='Interrogate type', choices=interrogate_type_choices, value='None', type="value", visible=False)
        interrogate_position = gr.Radio(label='Interrogate position', choices=['start', 'end'], value='end', type="value", visible=False)


        return [info, overlap, upscaler_index, fix_seed, interrogate_type, interrogate_position]

    def run(self, p, _, overlap, upscaler_index, fix_seed, interrogate_type, interrogate_position):
        processing.fix_seed(p)
        upscaler = shared.sd_upscalers[upscaler_index]

        p.extra_generation_params["SD upscale overlap"] = overlap
        p.extra_generation_params["SD upscale upscaler"] = upscaler.name

        initial_info = None
        seed = p.seed

        init_img = p.init_images[0]
        
        if(upscaler.name != "None"): 
            img = upscaler.scaler.upscale(init_img, 2, upscaler.data_path)
        else:
            img = init_img

        devices.torch_gc()

        grid = images.split_grid(img, tile_w=p.width, tile_h=p.height, overlap=overlap)

        batch_size = p.batch_size
        upscale_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []
        work_prompts = []

        for y, h, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])
                if interrogate_type == 'CLIP':
                    prompt = shared.interrogator.interrogate(tiledata[2])
                elif interrogate_type == 'DeepBooru':
                    prompt = get_deepbooru_tags(tiledata[2])
                else:
                    prompt = ''
                
                if prompt == '':
                    work_prompts.append(f"{p.prompt}")                
                elif p.prompt == '':
                    work_prompts.append(f"{prompt}")
                elif interrogate_position == 'start':
                    work_prompts.append(f"{prompt}, {p.prompt}")
                else:
                    work_prompts.append(f"{p.prompt}, {prompt}")
                print(work_prompts[-1])

        batch_count = math.ceil(len(work) / batch_size)
        state.job_count = batch_count * upscale_count

        print(f"SD upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} per upscale in a total of {state.job_count} batches.")

        result_images = []
        for n in range(upscale_count):
            if fix_seed:
                start_seed = seed
            else:
                start_seed = seed + n
            p.seed = start_seed

            work_results = []
            for i in range(batch_count):
                p.batch_size = batch_size
                p.init_images = work[i*batch_size:(i+1)*batch_size]
                prompt_bak = p.prompt
                p.prompt = work_prompts[i*batch_size:(i+1)*batch_size]

                state.job = f"Batch {i + 1 + n * batch_count} out of {state.job_count}"
                processed = processing.process_images(p)
                p.prompt = prompt_bak

                if initial_info is None:
                    initial_info = processed.info

                if not fix_seed:
                    p.seed = processed.seed + 1
                work_results += processed.images

            image_index = 0
            for y, h, row in grid.tiles:
                for tiledata in row:
                    tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                    image_index += 1

            combined_image = images.combine_grid(grid)
            result_images.append(combined_image)

            if opts.samples_save:
                images.save_image(combined_image, p.outpath_samples, "", start_seed, p.prompt, opts.samples_format, info=initial_info, p=p)

        processed = Processed(p, result_images, seed, initial_info)

        return processed
