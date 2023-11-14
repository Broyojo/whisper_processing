# DAC - https://github.com/descriptinc/descript-audio-codec/tree/main

import argparse
import os

import dac
import torch
from accelerate import PartialState
from tqdm import tqdm

parser = argparse.ArgumentParser(description="A fast DAC inference script")

parser.add_argument(
    "--audio_path",
    required=True,
    type=str,
    help="Path of the audio file or directory of audio files to be encoded and quantized.",
)

parser.add_argument(
    "--code_path",
    required=False,
    default="codes/",
    type=str,
    help="Path to save the codes. (default: codes)",
)

parser.add_argument(
    "--model_name",
    required=False,
    type=str,
    default="44khz",
    choices=["44khz", "24khz", "16khz"]
)

parser.add_argument(
    "--win_duration",
    required=False,
    type=float,
    default=1.0,
    help="Window duration in seconds. (default: 1.0)"
)

parser.add_argument(
    "--normalize_db",
    required=False,
    type=float,
    default=-16,
    help="Normalize db. (default: -16)"
)

def main():
    args = parser.parse_args()
    
    if os.path.isdir(args.audio_path):
        audio_files = [os.path.join(args.audio_path, file) for file in os.listdir(args.audio_path)]
    else:
        audio_files = [args.audio_path]
    
    if not os.path.exists(args.code_path):
        os.makedirs(args.code_path)
    
    if torch.cuda.is_available():
        distributed_state = PartialState()
        
        model = dac.DAC.load(dac.utils.download(model_type=args.model_name)).to(distributed_state.device)
        
        num_gpus = torch.cuda.device_count()
        
        # get num gpu files at a time
        # just do one audio file per gpu right now
        for i in tqdm(range(0, len(audio_files), num_gpus)):
            batch = audio_files[i : i + num_gpus]
            
            # run inference on each gpu
            with distributed_state.split_between_processes(batch) as device_batch:
                with torch.no_grad():
                    x = model.compress(device_batch[0], win_duration=args.win_duration, normalize_db=args.normalize_db)
                x.save(os.path.join(args.code_path, os.path.basename(device_batch[0]).split(".")[0] + ".dac"))

    else:
        model = dac.DAC.load(dac.utils.download(model_type=args.model_name))
        
        batch_size = 1
        for i in tqdm(range(0, len(audio_files), batch_size)):
            batch = audio_files[i : i + batch_size]
            
            with torch.no_grad():
                x = model.compress(batch[0])
            x.save(os.path.join(args.code_path, os.path.basename(batch[0]).split(".")[0] + ".dac"))
    
if __name__ == "__main__":
    main()