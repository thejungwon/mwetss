CUDA_VISIBLE_DEVICES=2 python inference.py \
-c /home/stud/jungwon/alternative/voicefilter2/config/config4.yaml \
-e /home/stud/jungwon/alternative/voicefilter2/embedder.pt  \
-t /home/stud/jungwon/test_data4/test/000000-target.wav \
-m /home/stud/jungwon/test_data4/test/000000-mixed.wav \
-r /home/stud/jungwon/audio_data/dev-other/4831/29134/4831-29134-0017.wav  \
-o ./ \
--checkpoint_path /home/stud/jungwon/alternative/voicefilter3/chkpt/voicefilter-mse-fast2/chkpt_109000.pt
