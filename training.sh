nohup python -u -m src.model.training_binary \
  --dataset drugbank \
  --ckpt_2d dataset/drugbank/pretrained_2D_DrugBank.pth \
  --ckpt_3d dataset/drugbank/pretrained_3D_DrugBank.pt \
  --id_map_path dataset/drugbank/id_map.csv \
  --cache_2d_path dataset/drugbank/drug2d_cache_DrugBank.pt \
  --cache_3d_path dataset/drugbank/drug3d_cache_DrugBank.pt \
  --train_path dataset/drugbank/fold1/train.csv \
  --val_path dataset/drugbank/fold1/valid.csv \
  --test_path dataset/drugbank/fold1/test.csv \
  --epochs 200 \
  --batch_size 64 \
  --seed 123 \
  --fold 1 \
  --metric_best acc \
  --save_dir checkpoints_ddi \
  --run_name drugbank_binary_fold1 \
  > ddi_drugbank_binary_fold1.log 2>&1 &

nohup python -u -m src.model.training_multiclass \
  --ckpt_2d dataset/drugbank/pretrained_2D_DrugBank.pth \
  --ckpt_3d dataset/drugbank/pretrained_3D_DrugBank.pt \
  --id_map_path dataset/drugbank/id_map.csv \
  --cache_2d_path dataset/drugbank/drug2d_cache_DrugBank.pt \
  --cache_3d_path dataset/drugbank/drug3d_cache_DrugBank.pt \
  --train_path dataset/drugbank/fold1/train.csv \
  --val_path dataset/drugbank/fold1/valid.csv \
  --test_path dataset/drugbank/fold1/test.csv \
  --num_labels 86 \
  --epochs 200 \
  --batch_size 128 \
  --seed 123 \
  --fold 1 \
  --metric_best acc \
  --save_dir checkpoints_ddi \
  --run_name drugbank_multiclass_fold1 \
  > ddi_drugbank_multiclass_fold1.log 2>&1 &
