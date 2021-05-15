#!/usr/bin/env bash

# original training
python main.py train -t /train_set/ -v /val_set/

# triplet loss training
python main.py train -t /train_set/ -v /val_set/ --loss triplet --miner '<miner-type>'

# inference
python main.py inference -d /inference_set/ -r /checkpoint.pth.tar --inference-strategy '<inference-strategy>'

# inference with probability
python main.py inference -d /inference_set/ -r /checkpoint.pth.tar --inference-strategy '<inference-strategy>' --probability --fusion '<fusion-op>'

# validation
python main.py validation -d /val_set/ --c /checkpoints --loss '<loss-type>' --miner '<miner-type>'

# evaluation
python main.py evaluation -g /ground_truth_data -c /predicted_data