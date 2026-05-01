export CUDA_VISIBLE_DEVICES=0

python egoloc_2D_demo.py \
  --video_path ./video1.mp4 \
  --output_dir output \
  --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
  --sam_checkpoint Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
  --bert_base_uncased_path Grounded-Segment-Anything/bert-base-uncased/ \
  --text_prompt hand \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --device cuda \
  --credentials auth.env \
  --action "Grasping the object" \
  --grid_size 3 \
  --max_feedbacks 1