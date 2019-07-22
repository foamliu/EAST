cd data/result/epoch_0_gt/
zip -r submit.zip *.txt
cd ../../..
mv data/result/epoch_0_gt/submit.zip ./script_test_ch4
cd script_test_ch4
python3 script.py -g=gt.zip -s=submit.zip
cd ..