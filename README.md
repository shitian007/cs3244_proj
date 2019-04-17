# Urban Sounds Classification

## Introduction
Preprocessing for [Urban8KSound](https://urbansounddataset.weebly.com/urbansound8k.html) dataset for custom training using [Tensorflow Audio Recognition](https://www.tensorflow.org/tutorials/sequences/audio_recognition#running_the_model_in_an_android_app).

## For convenience:

Training after creation of directory with labels:
```
python tensorflow/examples/speech_commands/train.py --data_url= \
--silence_percentage=10.0 \
--quantize=True \
--data_dir=<DATA_DIR>/UrbanSound8K/preprocessed \
--wanted_words=air_conditioner,car_horn,children_playing,dog_bark,drilling,engine_idling,gun_shot,jackhammer,siren,street_music \
--start_checkpoint=/tmp/speech_commands_train/conv.ckpt-5000
```

Creating the frozen graph for deployment:
```
python tensorflow/examples/speech_commands/freeze.py --start_checkpoint=/tmp/speech_commands_train/conv.ckpt-5000 \
--quantize=True \
--output_file=<OUTPUT_DIR>/UrbanSound8K/my_frozen_graph.pb
```

## Pre-built Frozen Graph
A pre-trained quantized frozen graph (urban_sounds.pb) at 70% validation-accuracy and the corresponding Android Application is included.
