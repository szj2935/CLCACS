### Dataset
The XLCoST dataset can be downloaded [here](https://drive.google.com/file/d/1tZfsYQgWmc2gG340ru5VbrZ5aLIZ41_6/view).
We only use the program-level data. The XLCoST_data folder should be placed under the current directory.

### Dependency
You can install dependencies using the following command.
```shell
pip install torch
pip install transformers
pip install opendelta
```
### Fine-tuning
We fine-tuned the model on a 3090 GPU. We provide scripts for fine-tuning with our method.
#### Stage One
```shell
sh run_stage_one.sh
```
#### Stage Two
```shell
sh run_stage_two.sh
```