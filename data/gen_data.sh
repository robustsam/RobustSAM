echo " "
echo " ##############################################################################################"
echo "=> Start generating degraded images of train dataset <=="
echo " ##############################################################################################"
echo " "
python augment.py --data_dir all_data/train

echo " "
echo " ##############################################################################################"
echo "=> Start generating degraded images of val dataset <=="
echo " ##############################################################################################"
echo " "
python augment.py --data_dir all_data/val

echo " "
echo " ##############################################################################################"
echo "=> Start generating degraded images of test dataset <=="
echo " ##############################################################################################"
echo " "
python augment.py --data_dir all_data/test

echo " "
echo " ##############################################################################################"
echo "=> Start generating degraded images of extra lvis dataset <=="
echo " ##############################################################################################"
echo " "
python augment.py --data_dir all_data/lvis

echo " "
echo " ##############################################################################################"
echo "=> Start generating degraded images of extra coco dataset <=="
echo " ##############################################################################################"
echo " "
python augment.py --data_dir all_data/coco

cd ..