import argparse
import json
import os
import threading
from queue import Queue

import librosa
import numpy as np
import torch
import torchaudio
from accelerate import PartialState
from tqdm import tqdm
from transformers import AutoProcessor, WavLMModel, pipeline

parser = argparse.ArgumentParser(description="A fast WavLM clustering script")

parser.add_argument(
    "--audio_path",
    required=True,
    type=str,
    help="Path of the audio file or directory of audio files to be transcribed.",
)

parser.add_argument(
    "--embeddings_path",
    required=False,
    default="embeddings/",
    type=str,
    help="Path to save the embedding output. (default: embeddings)",
)

parser.add_argument(
    "--model_name",
    required=False,
    default="patrickvonplaten/wavlm-libri-clean-100h-base-plus",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: patrickvonplaten/wavlm-libri-clean-100h-base-plus)",
)

parser.add_argument(
    "--batch_size",
    required=False,
    type=int,
    default=1,
    help="Number of parallel samples you want to compute. Reduce if you face OOMs. (default: 1)",
)

def main():
    args = parser.parse_args()
    
    if os.path.isdir(args.audio_path):
        audio_files = [os.path.join(args.audio_path, file) for file in os.listdir(args.audio_path)]
    else:
        audio_files = [args.audio_path]
    
    if not os.path.exists(args.embeddings_path):
        os.makedirs(args.embeddings_path)
    
    if False: # torch.cuda.is_available():
        pass
    else:
        # from datasets import load_dataset

        # dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
        # dataset = dataset.sort("id")
        # sampling_rate = dataset.features["audio"].sampling_rate
        
        # print(dataset[0]["audio"]["array"], dataset[0]["audio"]["array"].shape)
        
        # return

        model = WavLMModel.from_pretrained(args.model_name, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(args.model_name)
        
        print("running")
        
        for i in tqdm(range(0, len(audio_files), args.batch_size)):
            batch = []
            for audio_file in audio_files[i:i+args.batch_size]:
                audio, _ = librosa.load(audio_file, sr=16000)
                print(audio)
                batch.append(audio)
            
            inputs = processor(batch, return_tensors="pt", padding=True, sampling_rate=16000)
            
            with torch.no_grad():
                embeddings = model(**inputs)
                print(embeddings)
            
            # save embeddings as npz
            # for j in range(len(batch)):
            #     np.savez_compressed(os.path.join(args.embeddings_path, 
            #                                      f"{os.path.basename(batch[j]).split('.')[0]}_embedding"), 
            #                         outputs[0][j].cpu().numpy())
    
    # from datasets import load_dataset

    # dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    # dataset = dataset.sort("id")
    # sampling_rate = dataset.features["audio"].sampling_rate
    
    # processor = AutoProcessor.from_pretrained(args.model_name)
    # model = WavLMModel.from_pretrained(args.model_name, low_cpu_mem_usage=True)

    # inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    
    # print(sampling_rate)
    
    # print(inputs["input_values"].shape)
    
    # with torch.no_grad():
    #     outputs = model(**inputs)
    
    # print(outputs[0].shape)

if __name__ == "__main__":
    main()