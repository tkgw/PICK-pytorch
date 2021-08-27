#! /bin/sh
python \
  test.py \
  --checkpoint saved/models/PICK_Default/test_0824_154825/model_best.pth \
  --boxes_transcripts data/test_data_example/boxes_and_transcripts \
  --images_path data/test_data_example/images \
  --output_folder output/ \
  --gpu 0 --batch_size 2
