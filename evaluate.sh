#!/bin/bash

# shellcheck disable=SC2207
array=($(ls /home/hynek/skola/FEL/5.\ semestr/test/480p/))
k=("1" "5" "10" "15" "25" "50" "150" "250" "500" "1000" "1500" "2000" "-1")
for folder in "${array[@]}";
do
  rm features.npz
  rm similarity.npz
  for K in "${k[@]}";
  do
    echo "$folder - $K"
    python3.8 equation_3.py -K "$K" -d ~/skola/FEL/5.\ semestr/test/480p/"$folder"/ -a ~/skola/FEL/5.\ semestr/test/annot/"$folder"/ -c "/home/hynek/skola/FEL/5. semestr/semi-supervised-VOS/checkpoint-epoch-24-contrastive.pth.tar" -s ~/VOS_saves/contrastive/"$folder"
  done
done
