#! /bin/bash
echo -e "\033[32m creating conda env \033[0m"
cat ~/.condarc
CONDA_DIR="$(conda info --base)"
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda deactivate
conda remove -n pex --all -y
conda create -y -n pex python=3.8
conda activate pex

echo -e "\033[32m installing packages \033[0m"
# conda install pytorch=1.10.2 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge tape_proteins=0.5 -y
pip install -r requirements.txt -i https://pypi.python.org/simple/
cd stable_baselines3
pip install -e . -i https://pypi.python.org/simple/
cd ..

echo -e "\033[35m Install a lower version of torch for compatibility \033[0m"
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

echo -en "\033[35m cuda.is_available(): "
python -c "import torch; print(torch.cuda.is_available())"
pip list
echo -e "\033[0m"

echo -e "\033[32m downloading landscapes \033[0m"
if unzip -h; then echo; else sudo apt install unzip; fi
FILE="landscape_params.zip"
rm -r landscape_params
rm $FILE
ID="1uy9zgtJ60Z83LCbm7Z_AoksAkmK4CcsC"
URL="https://docs.google.com/uc?export=download&id=$ID"
COOKIES="./cookies.txt"
wget --load-cookies $COOKIES "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies $COOKIES --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$ID" -O $FILE && rm -rf $COOKIES

echo -e "\033[32m unzipping landscapes \033[0m"
unzip $FILE && rm $FILE