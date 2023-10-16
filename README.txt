YOLOv7-tiny模型描述檔：/yolov7_qat/cfg/training/yolov7-tiny.yaml
YOLOv3-tiny模型描述檔：/yolov7_qat/cfg/training/yolov3-tiny.yaml
訓練超參數：/yolov7_qat/data/
訓練結果：/yolov7_qat/runs/training/
模型檔：/yolov7_qat/runs/training/weights

訓練YOLOv7-tiny: python train.py --workers 8 --device 0 --batch-size 16 --data data/VOC.yaml --img 416 416 --cfg cfg/training/yolov7-tiny.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
訓練YOLOv3-tiny: python train.py --workers 8 --device 0 --batch-size 16 --data data/VOC.yaml --img 416 416 --cfg cfg/training/yolov3-tiny.yaml --weights '' --name yolov3 --hyp data/hyp.scratch.p5.yaml