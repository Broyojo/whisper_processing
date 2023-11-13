import argparse

import torch
from audiotools import AudioSignal

import dac

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

def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = dac.DAC.load(dac.utils.download(model_type=args.model_name))
    
    model.to(device)
    
    print(model)
    
if __name__ == "__main__":
    main()