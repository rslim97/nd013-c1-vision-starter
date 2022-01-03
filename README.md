# Object Detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission Template

### Project overview
In this project, we make use of tensorflow object detection API to train a custom object detector. All the training and testing dataset were from the [Waymo Open dataset](https://waymo.com/open/). This project was done on a local machine with NVIDIA GeForce GTX 1050 on a docker container. Steps to build and setting up the docker container are described in `build`. Next, we will go through the pipeline of training a tensorflow model including exploratory data analysis, splitting the dataset, model training, model improvement, and exporting a trained model. Finally, a short video for our prediction will be generated to showcase the result.

### Structure
This project is organized as follows:
* `build` contains instructions to build the docker image and to run a container.
* `download_process.py` will download original tf record files to `data/raw` before processing these files and saving them into `data/processed`. After `create_splits.py` is run, processed data will be moved into `train`,`val`, and `test` with ratios explained in a later section.
```
data
|--processed
|--raw
|--test
|--train
|--val
```
* The `pretrained_model` directory contains our downloaded SSD tf model. The `reference` directory contains `pipeline_new.config`, checkpoint files, `eval` and `train` folders containing evaluation and training events files respectively, `exported` directory containing a saved model. A similar structure applies to `improved` directory.
```
experiments
|--improved
|--pretrained_model
|--reference
|--exporter_main_v2.py
|--label_map.pbtxt
|--model_main_tf2.py
```
_* `model_main_tf2.py` to train or evaluate a model.
_* `exporter_main_v2.py` to convert trained model to a more compact inference model.
_* `label_map.pbtxt` 
* `video` folder stores both resultant inference video from both reference model and improved model.
* `plots` contains tensorboard plots.
* `images` contains images for this README.md file

the files directly under project folder:
* `download_process.py`
* `create_splits.py`
* `edit_config.py`
* `inference_video.py`
* `utils.py` 
* `Exploratory Data Analysis.ipynb`
* `Explore augmentations.ipynb`

### Set up
#### Download dataset
The dataset is avaiable under as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We will download the a portion of the files from cloud (not manually). To download the dataset:

1) Build a docker image as instructed in `build` folder. Run a docker container by running (modify based on your username):
`docker run --shm-size=256m --gpus all -v /home/reinaldy/Documents/nd013-c1-vision-starter:/app/project/ --network=host -ti project-dev bash`

2) Install gcloud:
`curl https://sdk.cloud.google.com | bash`
authorize login:
`gcloud auth login`

Before downloading, we need to make sure necessary libraries are installed:
```
pip install pyparsing==2.4.2
pip install avro-python3==1.10.0
pip install matplotlib==3.1.1
pip install absl-py==0.13
```
3) try downloading in small batch first:
```
CUDA_VISIBLE_DEVICES=0
python download_process.py --data_dir /app/project/data --size 5
```
make sure that 5 tf record files has been downloaded. 

4) Finally, run:
`python download_process.py --data_dir /app/project/data`
which will download 100 tf records to `/app/project/data/raw` directory. This will take some time.

For this project we only need a subset of the data provided (for example, we do not need the Lidar data). Therefore, we are going to download and trim immediately for each file. The `download_process.py` script will download original tf record files into `data/raw` and processed data will be stored in `data/processed`. Later, the raw data will be removed to save space.

```
def create_tf_example(filename, encoded_jpeg, annotations, resize=True):
    """
    This function create a tf.train.Example from the Waymo frame.

    args:
        - filename [str]: name of the image
        - encoded_jpeg [bytes]: jpeg encoded image
        - annotations [protobuf object]: bboxes and classes

    returns:
        - tf_example [tf.Train.Example]: tf example in the objection detection api format.
    """
    if not resize:
        # load the input encoded image into memory buffer using python BytesIO input/output library
        encoded_jpg_io = io.BytesIO(encoded_jpeg)
        image = Image.open(encoded_jpg_io)
        # extracts width and height information from this image
        width, height = image.size
        width_factor, height_factor = image.size
    else:
        # read the image as a tensor of uint8 type
        image_tensor = tf.io.decode_jpeg(encoded_jpeg)
        height_factor, width_factor, _ = image_tensor.shape
        image_res = tf.cast(tf.image.resize(image_tensor, (640, 640)), tf.uint8)
        encoded_jpeg = tf.io.encode_jpeg(image_res).numpy()
        width, height = 640, 640

    mapping = {1: 'vehicle', 2: 'pedestrian', 4: 'cyclist'}
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    filename = filename.encode('utf8')

    for ann in annotations:
        xmin, ymin = ann.box.center_x - 0.5 * ann.box.length, ann.box.center_y - 0.5 * ann.box.width
        xmax, ymax = ann.box.center_x + 0.5 * ann.box.length, ann.box.center_y + 0.5 * ann.box.width
        xmins.append(xmin / width_factor)
        xmaxs.append(xmax / width_factor)
        ymins.append(ymin / height_factor)
        ymaxs.append(ymax / height_factor)
        classes.append(ann.type)
        classes_text.append(mapping[ann.type].encode('utf8'))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpeg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example
```
#### Edit the config file
The Tf Object Detection API relies on config files. The config that we will use for reference is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. The [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) was placed under `app/project/experiments/pretrained_model`. The paper for Single Shot Detector can be read [here](https://arxiv.org/pdf/1512.02325.pdf). Other architectures can be viewed in the Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). 

We neet to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run:
`python edit_config.py --train_dir /app/project/data/train --eval_dir /app/project/data/val --batch_size 2 --checkpoint /app/project/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /app/project/experiments/label_map.pbtxt`

A new config file will be created, `pipeline_new.config`. This new config file can be placed to the `/app/project/experiments/reference` folder.

#### Model training and evaluation
To resume on a docker container previously created:
```
docker start  `docker ps -q -l` # restart it in the background
docker attach `docker ps -q -l` # reattach the terminal & stdin
```
To train the reference model, run:
```
python /app/project/experiments/model_main_tf2.py --model_dir=experiments/reference --pipeline_config_path=experiments/reference/pipeline_new.config
```
Open a new terminal and run:
```
docker exec -it <container-id> bash
```
where `<container-id>` is the id of the currently running container, which can be checked using:
`docker ps -a`
launch the evaluation process simultaneously with training process, by entering:
`CUDA_VISIBLE_DEVICES="" python /app/project/experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/`
To view tensorboard, open another new terminal running in the same docker id as described previously and enter:
`python -m tensorboard.main --logdir /app/project/experiments/reference/`

Improve the model by editing the the `pipeline_new.config` file. To do so in Sublime text editor, we need the power as a root user:
`sudo subl pipeline_new.config`
Similar training commands as the reference model can be used to training the improved model by replacing `reference` directory name to `improved` as detailed above.

#### Export model
After the training has finished, export the trained model by running:
`python /app/project/experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/`
Finally a short video of the model inference for any tf record file can be generated, as an example:
`python /app/project/inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /app/project/data/test/segment-10212406498497081993_5300_000_5320_000_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.mp4`

### Dataset
#### Dataset analysis
The dataset consists of urban and highway environment. In this project we will be focusing on three classes only, namely:
```
item {
    id: 1
    name: 'vehicle'
}

item {
    id: 2
    name: 'pedestrian'
}

item {
    id: 4
    name: 'cyclist'
}
```

as provided in `label_map.pbtxt`.
In the `Exploratory Data Analysis` notebook, we display camera images with the corresponding annotations. An example of such image is shown below:
<p align="center">
  <img src="images/eda.png" width="50%"/>
</p>
The codes for displaying images are as below:

```
dataset=get_dataset("/app/project/data/processed/*.tfrecord")
from matplotlib.patches import Rectangle
%matplotlib inline

def display_instances(batch):
    colormap={1:[1,0,0],2:[0,0,1],4:[0,1,0]}
    f,ax=plt.subplots(figsize=(8,8))
    img=batch["image"]
    w,h,_=img.shape
    bboxes=batch["groundtruth_boxes"]
    cls=batch["groundtruth_classes"]
    ax.imshow(img)
    for i in range(bboxes.shape[0]):
        bbox=bboxes[i]
        cl=cls[i]
        ymin=bbox[0]*h
        ymax=bbox[2]*h
        xmin=bbox[1]*w
        xmax=bbox[3]*w
        rect=Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,facecolor='none',edgecolor=colormap[cl])
        ax.add_patch(rect)
    ax.axis('off')
```

Additional insight to the dataset can be obtained through a function that displays information across tf records data struct, such as this:
<p align="center">
  <img src="images/additional_eda.png" width="25%"/>
</p>
#### Cross validation
We split the dataset to train, val, and test data by commonly used ratios of 0.8, 0.1, and 0.1 respectively. Because we are doing transfer learning, the number of training examples don't have to be huge. We shuffle the dataset before splitting, this is to ensure nearly equal distribution of scenes in the three resulting dataset. The validation performance of the trained model will be poor if the training and validation set have very different scenarios. We can monitor the evaluation loss and training loss using tensorboard. If the validation loss starts increasing while training loss decreases, it could be an indication of overfitting. The test split is used to test the model's accuracy.

### Training
#### Reference experiment
Tensorboard plots for the reference model are shown below:
<p align="center">
  <img src="plots/reference/1_loss" width="25%"/>
</p>
with a learning plotted below:
<p align="center">
  <img src="plots/reference/1_lr" width="25%"/>
</p>
we observed that as the learning rate increases from initial value of 0.013333 to about 0.04, the loss increases steeply. The loss decreases along with decaying learning rate.

<p align="center">
  <img src="plots/reference/1_precision" width="25%"/>
</p>
From the precision curve, we learn that the model did not learn anything useful until after 8000 steps. These prompts us to improve the reference model to increase its accuracy.

#### Improve on the reference
To improve the reference model, the following modifications were made:
1) Adjust the batch size
From experimentation, using a larger batch size generally results in faster decrease in loss.
2) Lower the learning rate
The learning rate is lowered from a base value of 0.04 to 0.002.
3) Random crop for data augmentation
As per the [SSD](https://arxiv.org/pdf/1512.02325.pdf) paper, random crop was utilized to increase the model's precision. Therefore, we try implementing ssd_random_crop as given in [preprocessor.proto](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto).
The resulting improved plots are given below:
1) Loss curves
<p align="center">
  <img src="plots/reference/2_loss" width="25%"/>
</p>
2) Precision
<p align="center">
  <img src="plots/reference/2_precision" width="25%"/>
</p>
3) Recall
<p align="center">
  <img src="plots/reference/2_recall" width="25%"/>
</p>
Comparisons can be made by viewing the inference videos available under `video`. The images below illustrate the improvement of inference accuracy.
<p align="center">
  <img src="images/reference" width="25%"/>
</p>
<p align="center">
  <img src="images/improved" width="25%"/>
</p>
