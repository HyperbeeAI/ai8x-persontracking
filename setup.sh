# Update-Install conda and create an environment.
sudo apt-get update
sudo apt-get install -y zip build-essential libgl1
if ! command -v conda &> /dev/null; then    
	curl -Lk https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh > miniconda_installer.sh
	chmod u+x miniconda_installer.sh
	bash miniconda_installer.sh -b -p ./conda -u
	conda/bin/conda init bash
	source conda/etc/profile.d/conda.sh
	conda config --set auto_activate_base false
	rm miniconda_installer.sh
fi	
CONDA_PATH=$(which conda)
export PATH="$(dirname $CONDA_PATH)/../bin:$PATH"
source ~/.bashrc
conda create -y -k --prefix ./venv_p39_nanotracker python=3.9
conda activate ./venv_p39_nanotracker/

# Install required libraries
pip install -r dependencies.txt

# Clone the Yet-Another repo and download weight.
cd ./efficientdet_comparison/
git clone --depth 1 https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
cd ./Yet-Another-EfficientDet-Pytorch/
mkdir weights
wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d0.pth -O weights/efficientdet-d0.pth
cd ../ 
# back to efficientdet_comparison/

# Download datasets (val_data.tar.gz & val_bbox.txt) 
## datasets are taken from codalab competition https://competitions.codalab.org/competitions/20132 
## note that we are not redistributing the dataset, we are only using it for evaluation purposes here
## you need to be registered to the competition first to download this, check dataset website for further info
mkdir data
cd ./data
mkdir original_wider
cd ./original_wider
gdown 1-nxbaSBK_JnFAOhIkdDPeG0Ho3_IJBq0
tar -xf val_data.tar.gz
rm val_data.tar.gz
gdown 1geyoI8mdk2lPH047ajAP65imxwaxm8la
cd ../../
# back to efficientdet_comparison/

# Copy wider2coco to Yet-Another repo and use to conver Wider Dataset in to COCO format.
python wider2coco.py -ip ./data/original_wider/val_data -af ./data/original_wider/val_bbox.txt 

# Copy yml file to projects.
cp ./wider.yml ./Yet-Another-EfficientDet-Pytorch/projects/

# Delete existing and copy our eval file
cp ./coco_eval_efficientdet.py ./Yet-Another-EfficientDet-Pytorch/
mv ./Yet-Another-EfficientDet-Pytorch/coco_eval_efficientdet.py ./Yet-Another-EfficientDet-Pytorch/coco_eval.py
