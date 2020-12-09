#!/bin/sh

tar -xvf data/meme_templates.tar -C data/
lrunzip -cdivv data/11-25-20_21-1500.tsv.lrz -o data/meme_data.tsv
lrunzip -cdivv data/gcc_full.tsv.lrz -O data/
lrunzip -cdivv data/gcc_captions.txt.lrz -O data/