# Dialogue-Reading-Comprehension

1. set environment

   1. install python3.6

      ```sh
      sudo apt-get update
      sudo apt-get install python3.6
      ```

   2. install pip for python3.6

      ```sh
      cd /home/ubuntu
      curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
      sudo python3.6 get-pip.py
      ```

   3. install some python packages requied

      ```sh
      pip3 install keras
      pip3 install tensorflow-gpu
      pip3 install keras-bert
      ```

   4. install cuda 10.0

      ```sh
      cd /home/ubuntu
      wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
      sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
      sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
      sudo apt-get update
      sudo apt-get install cuda-10-0
      reboot
      ```

   5. install cudnn

      Follow the instruction of https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#download to download the install file to local

      upload these files to server using "scp" and execute following commands in the server

      ```sh
      sudo dpkg -i libcudnn7_7.5.0.56-1+cuda10.0_amd64.deb
      sudo dpkg -i libcudnn7-dev_7.5.0.56-1+cuda10.0_amd64.deb
      sudo dpkg -i libcudnn7-doc_7.5.0.56-1+cuda10.0_amd64.deb
      ```

      Follow the instruction of https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#download to verify

2. Run

   ```sh
   cd /home/ubuntu
   git clone https://github.com/FrankLicm/Multivariable-character-Identification.git
   cd "Multivariable-character-Identification/SV&MVScode"
   wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
   unzip uncased_L-12_H-768_A-12.zip 
   rm -rf uncased_L-12_H-768_A-12.zip
   cd uncased_L-12_H-768_A-12
   mv * ..
   cd ..
   rm -rf uncased_L-12_H-768_A-12
   ```

   for task 1 training:

   every time you run, you can choose different random_seed

   ```sh
   python3.6 exp_bert.py --train_file ../data/sv-new/trn.json --dev_file ../data/sv-new/dev.json --config_path bert_config.json --model_path bert_model.ckpt --dict_path vocab.txt --model bert --logging_to_file log.txt --save_model model.h5 --stopwords stopwords.txt --learning_rate 2e-5  --random_seed 8237 --nb_epoch 8  --batch_size 6 --gpu 0 
   ```

   for task 1 testing:

   ```sh
   python3.6 exp_bert.py --train_file ../data/sv-new/trn.json --dev_file ../data/sv-new/tst.json --config_path bert_config.json --model_path bert_model.ckpt --dict_path vocab.txt --model bert --logging_to_file log.txt --pre_trained model.h5 --stopwords stopwords.txt --test_only True --gpu 0
   ```

   

   For other tasks just change the **train_file** and **dev_file**

   If you want to run the command in the backend, this will output error infomation to "log" file if there is a error occurring:

   ```sh
   nohup   your_original_command  >/dev/null 2>log &
   ```

3. some other useful commands

   1. check GPU status

      ```sh
      nvidia-smi
      ```

   2. check current running python program

      ```sh
      ps -ef | grep python
      ```

   3. copy file from local to aws

      ```sh
      scp -i "your pem" “your file path” your_sever_adderess:
      ```

      remember add ":"

   4. copy directory from local to aws

      ```sh
      scp -i "your pem" -r “your directory path” your_sever_adderess:
      ```

      remember add ":"

   5. check linux version

      ```sh
      uname -m && cat /etc/*release
      ```

      

      