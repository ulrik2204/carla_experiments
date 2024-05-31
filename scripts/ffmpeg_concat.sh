#! /bin/bash
# ls $1/*.png | sort -V > filelist.txt
# while read -r file; do echo "file '$PWD/$file'" >> input.txt; done < filelist.txt
ffmpeg -framerate 20 -pattern_type glob -i "$1/*.png" -vf "scale=1208:910" -c:v libx264 -pix_fmt yuv420p out.mp4
# ffmpeg -framerate 20 -f concat -safe 0 -i input.txt -vf "scale=1208:910" -c:v libx264 -pix_fmt yuv420p out.mp4
# ffmpeg -f concat -safe 0 -i filelist.txt -vf "scale=1208:910" -framerate 20 -c:v libx264 -pix_fmt yuv420p out.mp4



