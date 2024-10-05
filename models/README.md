Current Models:

- 2023-12-6: Final model produced through second round of training by Makai Mann at LL. Trained at a starting distance of 12464m, max position offset of 150m and max orientation offset of 15 degrees.

- 2024-3-12 and 2024-3-13: Models produced through training by Colette Scott at WPI, 3-13 being the final model to be used for data analysis and running the autolander scripts.  Trained at a starting distance of 12464m, max position offset of 150m and max orientation offset of 15 degrees. This model was generated to establish a model representative of WPI's system. Note: the camera position used during data generation is different between LL and WPI models.

- 2024-3-31 and 2024-4-1: Models produced through training by Colette Scott at WPI, 4-1 being the final model to be used for data analysis and running the autolander scripts.  Trained at a starting distance of 12464m, max position offset of 50m and max orientation offset of 10 degrees. This model was generated with the purpose of analyzing OOD Data. Note: the camera position used during data generation is different between LL and WPI models.

- 2024-7-11, 2024-7-12, 2024-7-12_og: Models produced through training by Ava Chadbourne at WPI. This was an initial attempt at training data using the 1500m maximum distance from runway 'safe' cone of data. Training was done incorrectly. Do not use for analysis or autolanding. 

- 2024-7-15_No_h, 2024-7-16_No_h: Models produced through training by Ava Chadbourne at WPI. These models were trained without using the height error in an attempt to increase crosstrack error accuracy. Models cannot succesfully autoland or analyze data. 

- 2024-7-16, 2024-7-17: Models produced through training by Ava Chadbourne at WPI, with 7-17 being the final model to be used for data analysis and running the autolander scripts. Trained at a starting distance of 1500m, max position offset of 50m and max orientation offset of 15 degrees. This model was generated with data generated within a 3.5deg 'safe cone' of the glideslope for the purposes of analyzing crosstrack error.

How to train:

Trained via transfer learning using Resnet50 as backbone. As I understand it 1st round backbone/all layers except final are frozen to allow for feature detection tuning. Second round unfrozen to allow tuning of remaining layers.

- Round 1: Simply run the train.py script as is- no need to set args as backbone is frozen by default.

- Round 2: Set the --start-model and --unfrozen parameters when runnning the script to set the start model as the one trained during the previous round and unfreeze the backbone. Example for 4-1 model:

    python train.py --start-model="/home/colette/XPlaneAutolandScenario/models/2024-3-3/best_model_params.pt" --unfreeze
