# Training and Inference

## Training

If you want to train the model, you can run the following command:

```bash
./tools/dist_train.sh ./projects/configs/prompttrack/f3_prompttrack.py 8 --work_dir=./work_dirs/f3_prompttrack
```

## Inference
If you want to inference the model, you can run the following command:

```bash
python ./tools/test_prompt_tracking.py ./projects/configs/prompttrack/f3_prompttrack.py ./work_dirs/f3_prompttrack/epoch_12.pth --eval=bbox --out=./work_dirs/f3_prompttrack/results.pkl --jsonfile_prefix=./work_dirs/f3_prompttrack/
```


## Visualization
If you want to visualize the results, you can run the following command:

```
python ./tools/visualization.py ./work_dirs/f3_prompttrack/results_prompt_tracking.json --show-dir=./work_dirs/f3_prompttrack/visualization --scene_token=scene/token --prompt_filename=path/to/prompt/filename
```