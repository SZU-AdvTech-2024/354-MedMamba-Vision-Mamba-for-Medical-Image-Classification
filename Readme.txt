Readme

Name: Chan Han Xiang
Student ID: 2411103003

1. Results
Contains results for Medmamba, MedViT and Resnet50, categorised by subfolder

For each model, six graphs across training and validation epochs are generated:
- Accuracy + AUC
- Loss + F1
- Sensitivity + Specificity

Text document to each model is also provided. Results to each model can be discerned from file name.

2. Train_scripts

Contains Python files to run training and validation epochs. Training scripts to each model can be discerned from file name.

2a. src

Contains base codes for models.
- MedMamba - base code for MedMamba
- Medvit - base code for Medvit
- medvit_utils - utilities code to Medvit.py

3. Test_scripts

Contains python files to run test scripts. Test scripts to each model can be discerned from file name.

To load trained weights, insert .pth file along this line:

pretrained_path = "xxx.pth"

4. Weights

Contains .pth files for Medmamba, MedViT and Resnet50, categorised by subfolder. Within each subfolder, pretrained weights applicable to each model type are marked within its name.

Only model weights listed in Table 3/4 are provided; weights generated from hyperparameter tuning are not provided considering space constraints

5. requirements.txt

Contains python packages used in Conda environment on Linux