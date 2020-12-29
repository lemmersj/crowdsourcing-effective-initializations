#!/bin/bash
# OTB2015
# Script from the DaSiamRPN repository located at 
# https://github.com/foolwood/DaSiamRPN
mkdir OTB2015 && cd OTB2015
baseurl="http://cvlab.hanyang.ac.kr/tracker_benchmark"
wget "$baseurl/datasets.html"
cat datasets.html | grep '\.zip' | sed -e 's/\.zip".*/.zip/' | sed -e s'/.*"//' >files.txt
cat files.txt | xargs -n 1 -P 8 -I {} wget -c "$baseurl/{}"
ls *.zip | xargs -n 1 unzip
rm -r __MACOSX/
cd ..
