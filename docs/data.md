# Data Files and Pretrained Models

## 1. Download nuScenes
Download the nuScenes dataset, soft-link it to ./data/nuscenes. This step is compulsory.

## 2. Download Preprocessed nuScenes Files [[Here](https://tri-ml-public.s3.amazonaws.com/github/pftrack/data.zip)]

The generation of preprocessed nuScenes files can be found in [PF-Track](https://github.com/TRI-ML/PF-Track/blob/main/documents/preprocessing.md). 

## 3. Download NuPrompt Dataset [[Here](https://github.com/wudongming97/Prompt4Driving/releases/download/v1.0/nuprompt_v1.0.zip)]

Each json file comprises four parts:

```
{
    'scene_token':                                                    <str> -- unique identifier of a scene
    'prompt':                                                               <str> -- language prompt
    'frame_token_object_token':{
        'frame_token_1':[                                          <str> -- unique identifier of a frame
            'instance_token_1',                                  <str> -- unique identifier of an object
            'instance_token_2',         
            ...
        ]        
        'frame_token_2':[               
            'instance_token_1',         
            'instance_token_3',         
            ...
        ]            
        ...
    }
    'original_prompt':                                               <str> -- original language prompt
},
```

## 4. Download Preprocessed NuPrompt Val Files [[Here](https://github.com/wudongming97/Prompt4Driving/releases/download/v1.0/nuprompt_infos_val.pkl)] [[Here](https://github.com/wudongming97/Prompt4Driving/releases/download/v1.0/instance_token_to_id_map.pkl)]

Different from the original nuScenes dataset that inference each scene once, NuPrompt dataset requires testing each prompt, which is time-consuming.
To save time, please download the preprocessed val file `nuprompt_infos_val.pkl`.
It contains two random prompts for each scene in the validation set, and the corresponding generation script can be found in `./tools/create_nuprompt_infos_val.py`.
Besides, the other file `instance_token_to_id_map.pkl` should be downloaded for easy validation. 

After doing so, the structure of that directory will be:


```txt
- nuscenes
    - v1.0-mini
    - v1.0-test
    - v1.0-trainval
    ...
    - tracking_forecasting_infos_test.pkl
    - tracking_forecasting_infos_train.pkl
    - tracking_forecasting_infos_val.pkl
    - tracking_forecasting-mini_infos_train.pkl
    - tracking_forecasting-mini_infos_val.pkl
    ...
    - nuprompt_v1.0
    - nuprompt_infos_val.pkl
    - instance_token_to_id_map.pkl
```

## 5. Download Single-frame Detection Model [[Here](https://tri-ml-public.s3.amazonaws.com/github/pftrack/f1.zip)]

After downloading, please extract them to `./ckpts/` of the root directory of this repository. The single-frame detectors is provided from the first stage of training PF-Track. Downloading them will save the effort in training single-frame detectors for reproducing our results.