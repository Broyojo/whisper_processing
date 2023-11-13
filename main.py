# DDP - https://huggingface.co/docs/diffusers/training/distributed_inference

import argparse
import json
import os
import threading
from queue import Queue

import torch
from accelerate import PartialState
from tqdm import tqdm
from transformers import pipeline

parser = argparse.ArgumentParser(description="A fast SST inference script")

parser.add_argument(
    "--audio_path",
    required=True,
    type=str,
    help="Path of the audio file or directory of audio files to be transcribed.",
)

parser.add_argument(
    "--transcript_path",
    required=False,
    default="output.jsonl",
    type=str,
    help="Path to save the transcription output. (default: output.jsonl)",
)

parser.add_argument(
    "--model_name",
    required=False,
    default="distil-whisper/distil-medium.en",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: distil-whisper/distil-medium.en)",
)

parser.add_argument(
    "--chunk_batch_size",
    required=False,
    type=int,
    default=24,
    help="Number of parallel chunk batches you want to compute. Reduce if you face OOMs. (default: 24)",
)

parser.add_argument(
    "--batch_size",
    required=False,
    type=int,
    default=1,
    help="Number of parallel samples you want to compute. Reduce if you face OOMs. (default: 1)",
)

parser.add_argument(
    "--flash",
    required=False,
    type=bool,
    default=False,
    help="Whether to use Flash Attention 2 for inference. (default: False)",
)

parser.add_argument(
    "--timestamp",
    required=False,
    type=str,
    default=False,
    choices=["chunk", "word"],
    help="Whether to use chunk-level timestamps or word-level timestamps. (default: None)",
)

parser.add_argument(
    "--chunk_length_s",
    required=False,
    type=float,
    default=30.0,
    help="Length of each chunk in seconds. (default: 30.0)",
)

def main():
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        distributed_state = PartialState()
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model_name,
            torch_dtype=torch.float16,
            model_kwargs={"use_flash_attention_2": args.flash, "low_cpu_mem_usage": True},
            device=distributed_state.device,
        )
        
        done_queue = Queue()
        
        def write_to_jsonl():
            while True:
                outputs = done_queue.get()
                if outputs == None:
                    break

                with open(args.transcript_path, "a") as f:
                    json.dump(outputs, f)
                    f.write("\n")
        
        # each gpu runs it's own copy of the model
        if os.path.isdir(args.audio_path):
            audio_files = [os.path.join(args.audio_path, file) for file in os.listdir(args.audio_path)]
        else:
            audio_files = [args.audio_path]
        
        # start writer thread
        writer_thread = threading.Thread(target=write_to_jsonl)
        writer_thread.start()
        
        num_gpus = torch.cuda.device_count()
        
        # get num gpu files at a time
        for i in tqdm(range(0, len(audio_files), num_gpus * args.batch_size)):
            batch = audio_files[i : i + num_gpus * args.batch_size]
            
            # run inference on each gpu
            with distributed_state.split_between_processes(batch) as device_batch:
                outputs = pipe(
                    device_batch,
                    chunk_length_s=args.chunk_length_s,
                    batch_size=args.chunk_batch_size,
                    return_timestamps=args.timestamp,
                )
                
                for output, b in zip(outputs, batch):
                    output["id"] = b.split(".")[0]
                    done_queue.put(output)

        done_queue.put(None)
        writer_thread.join()
        
    else:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model_name,
            torch_dtype=torch.float32,
            model_kwargs={"low_cpu_mem_usage": True},
        )
        pipe.model = pipe.model.to_bettertransformer()

        if os.path.isdir(args.audio_path):
            audio_files = [os.path.join(args.audio_path, file) for file in os.listdir(args.audio_path)]
        else:
            audio_files = [args.audio_path]
       
        for i in tqdm(range(0, len(audio_files), args.batch_size)):
            batch = audio_files[i : i + args.batch_size]

            outputs = pipe(
                batch,
                chunk_length_s=args.chunk_length_s,
                batch_size=args.chunk_batch_size,
                return_timestamps=args.timestamp,
            )
            
            # write to jsonl
            with open(args.transcript_path, "a") as f:
                for output, b in zip(outputs, batch):
                    output["id"] = b.split(".")[0]
                    json.dump(output, f)
                    f.write("\n")
    
if __name__ == "__main__":
    main()