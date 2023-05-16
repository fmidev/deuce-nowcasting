# How to use the configuration

1. Copy the default configuration in config/default to a new config/current folder
2. Modify the config to your needs. 
    - Remember to link the platform.yaml file to one that contains the platform specific variables where you're running the code.
3. Run the training/prediction/etc .. using the "current" config, for example with `python train_model.py config/current`
4. The configuration used will be automatically copied to your results folder, into a subfolder corresponding to the ongoing experiment 
