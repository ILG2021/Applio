"""Inference-only deployment entry point.

Models are discovered below ``models/<model-folder>/``.  A model folder may
contain one or more ``.pth`` weights, an ``.index`` file, and an optional
``speakers.json``/``speaker.json``/``speaker_config.json`` file.
"""

import argparse
import os
import shutil
import sys


now_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(now_dir)
sys.path.insert(0, now_dir)

# Keep the deployment entry point local and telemetry-free.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["APPLIO_INFER_DEPLOYMENT"] = "1"
os.environ.setdefault("APPLIO_INFER_MODEL_ROOT", os.path.join(now_dir, "models"))
os.makedirs(os.environ["APPLIO_INFER_MODEL_ROOT"], exist_ok=True)

config_path = os.path.join(now_dir, "assets", "config.json")
config_template_path = os.path.join(now_dir, "assets", "config_template.json")
if not os.path.exists(config_path):
    shutil.copy(config_template_path, config_path)

import gradio as gr

from rvc.lib.platform import platform_config

platform_config()

from tabs.inference import inference as inference_module
from core import run_prerequisites_script

run_prerequisites_script(pretraineds_hifigan=True, models=True, exe=True)


def run_infer_with_named_speaker(*args):
    values = list(args)
    if len(values) > 7:
        values[-1] = inference_module._speaker_id(values[7], values[-1])
        values[7] = inference_module.resolve_model_path(values[7])
    return inference_module._run_infer_script(*values)


def run_batch_infer_with_named_speaker(*args):
    values = list(args)
    if len(values) > 7:
        values[-1] = inference_module._speaker_id(values[7], values[-1])
        values[7] = inference_module.resolve_model_path(values[7])
    return inference_module._run_batch_infer_script(*values)


# enforce_terms() resolves these names from the inference module at runtime.
inference_module._run_infer_script = inference_module.run_infer_script
inference_module._run_batch_infer_script = inference_module.run_batch_infer_script
inference_module.run_infer_script = run_infer_with_named_speaker
inference_module.run_batch_infer_script = run_batch_infer_with_named_speaker


with gr.Blocks(title="Applio Inference") as applio_infer:
    gr.Markdown("# Applio Inference")
    gr.Markdown("Inference-only deployment")
    inference_module.inference_tab()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Applio inference UI.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6969)
    parser.add_argument(
        "--root-path",
        default=None,
        help="URL mount path or external root URL used by the reverse proxy.",
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    applio_infer.launch(
        server_name=args.server_name,
        server_port=args.port,
        root_path=args.root_path,
        share=args.share,
        inbrowser=args.open,
    )
