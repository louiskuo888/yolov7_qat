for i in $(seq 1 10)
do
  python train.py --workers 8 --device 0 --batch-size 16 --data data/VOC.yaml --img 416 416 --cfg cfg/training/yolov7-tiny.yaml --weights '' --name yolov7_p5leaky --hyp data/hyp.scratch.p5.yaml &&
 sleep 5
done