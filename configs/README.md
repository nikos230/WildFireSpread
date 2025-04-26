## Configs for Training / Tesing the Models

### train_test_unet2d.yaml / train_test_unet3d.yaml
This config is used for UNet2D model and for UNet2D Baseline model. You can chnage every setting you want. It is important to specify path for checkpoints to be saved in: `checkpoints:` after each epoch of training. Also in `checkpoint_path:`
you have to specify best model path from training or use the pre-trained model checkpoint. Finaly if you run xAI_test_unet3d.py you have to specify a path for xAI stuff to be saved in `save_results_path:`

### dataset.yaml
This config is used for the dataset samples. After you download the dataset and extract it you have to specify the path in 'corrected_dataset_path:' or if you sample from the mesogeos datacube and want to correct it before training specify the path to 
`dataset_path:` and if you change the y, x or time dimentions update the `reference_dims:` which is used as baseline for when the samples are been corrected <br />

From this config the model can be configured to train on specify years and countries. By default its configured to train on all years and all countries, execpt years 2021 and 2022 the first is used for validation while training and the second is used for final testing.
Also it can be configured to exclude countries from training or tesing, and configured to test on a specfic country only. Finaly it can include fire only bigger or smaller than a specify size in hectare (ha) this can only be used if both bigger and smaller size is used
together like 150 < fire size < 500, so the fires will be between 150 and 500 ha, it can not be used only with < 150 or > 500.
