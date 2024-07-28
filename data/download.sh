mkdir all_data
cd all_data

echo " "
echo " ##############################################################################################"
echo "=> Start downloading train dataset (consists of image from LVIS, MSRA10K and ThinObject-5k) <=="
echo " ##############################################################################################"
echo " "
mkdir train
cd train
wget https://huggingface.co/robustsam/robustsam/resolve/main/dataset/train/clear.zip
unzip clear.zip
wget https://huggingface.co/robustsam/robustsam/resolve/main/dataset/train/masks.zip
unzip masks.zip
cd ..

echo " "
echo " ##############################################################################################"
echo "=> Start downloading val dataset (consists of image from LVIS, MSRA10K and ThinObject-5k) <=="
echo " ##############################################################################################"
echo " "
mkdir val
cd val
wget https://huggingface.co/robustsam/robustsam/resolve/main/dataset/val/clear.zip
unzip clear.zip
wget https://huggingface.co/robustsam/robustsam/resolve/main/dataset/val/masks.zip
unzip masks.zip
cd ..

echo " "
echo " ##############################################################################################"
echo "=> Start downloading test dataset (consists of image from MSRA, NDD20, STREETS and FSS-1000) <=="
echo " ##############################################################################################"
echo " "
mkdir test
cd test
wget https://huggingface.co/robustsam/robustsam/resolve/main/dataset/test/clear.zip
unzip clear.zip
wget https://huggingface.co/robustsam/robustsam/resolve/main/dataset/test/masks.zip
unzip masks.zip
cd ..

echo " "
echo " ##############################################################################################"
echo "=> Start downloading extra COCO dataset <=="
echo " ##############################################################################################"
echo " "
mkdir coco
cd coco
wget https://huggingface.co/robustsam/robustsam/resolve/main/dataset/coco/clear.zip
unzip clear.zip
wget https://huggingface.co/robustsam/robustsam/resolve/main/dataset/coco/masks.zip
unzip masks.zip
cd ..

echo " "
echo " ##############################################################################################"
echo "=> Start downloading extra LVIS dataset <=="
echo " ##############################################################################################"
echo " "
mkdir lvis
cd lvis
wget https://huggingface.co/robustsam/robustsam/resolve/main/dataset/lvis/clear.zip
unzip clear.zip
wget https://huggingface.co/robustsam/robustsam/resolve/main/dataset/lvis/masks.zip
unzip masks.zip
cd ..

cd ..