# Update-Install conda and create an environment.
sudo apt-get update
sudo apt-get install -y zip build-essential libgl1
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip
wget -L https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -b -p ~/miniconda3
rm Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
echo 'export PATH="~/miniconda3/bin:$PATH"' >> ~/.bashrc 
source ~/.bashrc
conda update conda -y
conda create -p ~/venv_p39_nanotracker/ python=3.9 -y
source activate ~/venv_p39_nanotracker/

# Install required libraries
pip install -r dependencies.txt

# Clone the Yet-Another repo and download weight.
cd ../
git clone --depth 1 https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
cd ./Yet-Another-EfficientDet-Pytorch/
mkdir weights
wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d0.pth -O weights/efficientdet-d0.pth

# Download datasets (val_data.tar.gz & val_bbox.txt) 
## Datasets are taken from codalab competition https://competitions.codalab.org/competitions/20132 
mkdir data
cd ./data
mkdir original_wider
cd ./original_wider
gdown 1-nxbaSBK_JnFAOhIkdDPeG0Ho3_IJBq0
tar -xf val_data.tar.gz
rm val_data.tar.gz
gdown 1geyoI8mdk2lPH047ajAP65imxwaxm8la
cd ../../

# Copy wider2coco to Yet-Another repo and use to conver Wider Dataset in to COCO format.
cp ../wider2coco.py ./
python wider2coco.py -ip ./data/original_wider/val_data -af ./data/original_wider/val_bbox.txt 

# Copy yml file to projects.
cp ../install_evaluations/wider.yml ./projects/

# Delete existing and copy our eval file
rm ./coco_eval.py
cp ../install_evaluations/coco_eval_efficientdet.py ./
mv coco_eval_efficientdet.py coco_eval.py

# Now the evaluation can be runned by "conda activate ~/venv_p39_nanotracker & python coco_eval.py -p wider -c 0 -w ./weights/efficientdet-d0.pth"
cd ../
mv wider2coco.py ../
mv coco_eval.py ../
cp -r Yet-Another-EfficientDet-Pytorch/data/ ../data/
cp -r Yet-Another-EfficientDet-Pytorch/datasets/ ../datasets/