from dataclasses import dataclass

from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules import script_callbacks
import modules.scripts as scripts
from modules import shared
import gradio as gr

VERSION = 'v0.1.3'

# luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B
# LUTS: [-K, -M, C, Y]

LUTS = [0.0, 0.0, 0.0, 0.0]

# https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/configs/v1-inference.yaml#L17
# (1.0 / 0.18215) / 2 = 2.74499039253
# (1.0 / 0.13025) / 2 = 3.83877159309

DYNAMIC_RANGE = [3.839, 2.745, 2.745, 2.745]


def normalize_tensor(x, r):
    x_min = abs(float(x.min()))
    x_max = abs(float(x.max()))

    delta = (x_max - x_min) / 2.0
    x -= delta

    ratio = r / float(x.max())

    if ratio > 0.95:
        x *= ratio

    return x + delta

# Maximize/normalize tensor
def maximize_tensor(input_tensor, boundary=4, channels=[0, 1, 2]):
    min_val = input_tensor.min()
    max_val = input_tensor.max()

    normalization_factor = boundary / max(abs(min_val), abs(max_val))
    input_tensor[0, channels] *= normalization_factor

    return input_tensor

original_callback = KDiffusionSampler.callback_state


def center_callback(self, d):
    options: "DiffusionCG.CGOptions" = self.diffcg_options

    if not options.is_enabled():
        return original_callback(self, d)

    batchSize = d[self.diffcg_tensor].size(0)
    for image_num in range(batchSize):
        for channel in range(4):

            if options.enable_centering:
                d[self.diffcg_tensor][image_num][channel] += options.channel_shift * (
                        LUTS[channel] - d[self.diffcg_tensor][image_num][channel].mean())

            # if options.enable_normalization and (d['i'] + 1) >= self.diffcg_last_step - 1:
            #     d[self.diffcg_tensor][image_num][channel] = normalize_tensor(d[self.diffcg_tensor][image_num][channel],
            #                                                                  DYNAMIC_RANGE[channel])
        if options.enable_normalization and (d['i'] + 1) >= self.diffcg_last_step - 1:
            d[self.diffcg_tensor][image_num] = maximize_tensor(d[self.diffcg_tensor][image_num])
    return original_callback(self, d)


KDiffusionSampler.callback_state = center_callback


class DiffusionCG(scripts.Script):
    @dataclass
    class CGOptions:
        enable_centering: bool
        enable_normalization: bool

        channel_shift: float
        full_tensor_shift: float

        def is_enabled(self):
            return self.enable_normalization or self.enable_centering

    options = CGOptions

    def title(self):
        return "DiffusionCG"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(f'Diffusion CG {VERSION}', open=False):
            enableG = gr.Checkbox(label="Enable (Global)")

            with gr.Row():
                with gr.Group():
                    gr.Markdown('<h3 align="center">Recenter</h3>')
                    enableC = gr.Checkbox(label="Enable")

                    channel_shift = gr.Slider(label="Channel shift", maximum=2.0)
                    full_tensor_shift = gr.Slider(label="Full tensor shift", maximum=2.0)

                with gr.Group():
                    gr.Markdown('<h3 align="center">Normalization</h3>')
                    enableN = gr.Checkbox(label="Enable")

        return [enableG, enableC, enableN, channel_shift, full_tensor_shift]

    def before_hr(self, p, *args):
        KDiffusionSampler.diffcg_options.enable_normalzation = False

    def process(self, p, enableG: bool, enableC: bool, enableN: bool, channel_shift: float, full_tensor_shift: float):
        self.options = DiffusionCG.CGOptions(
            enable_centering=enableC,
            enable_normalization=enableN,
            channel_shift=channel_shift,
            full_tensor_shift=full_tensor_shift
        )

        KDiffusionSampler.diffcg_options = self.options
        KDiffusionSampler.diffcg_enable = enableG
        KDiffusionSampler.diffcg_recenter = enableC
        KDiffusionSampler.diffcg_normalize = enableN
        KDiffusionSampler.diffcg_tensor = 'x' if p.sampler_name.strip() == 'Euler' else 'denoised'

        if not hasattr(p, 'enable_hr') and hasattr(p,
                                                   'denoising_strength') and not shared.opts.img2img_fix_steps and p.denoising_strength < 1.0:
            KDiffusionSampler.diffcg_last_step = int(p.steps * p.denoising_strength) + 1
        else:
            KDiffusionSampler.diffcg_last_step = p.steps


def restore_callback():
    KDiffusionSampler.callback_state = original_callback


script_callbacks.on_script_unloaded(restore_callback)
