import argparse  # 引入命令列參數解析的庫
import logging  # 引入日誌記錄庫
import math  # 引入數學庫
import os  # 引入操作系統的庫
import random  # 引入隨機數生成庫
import time  # 引入時間庫
from copy import deepcopy  # 引入深度複製的庫
from pathlib import Path  # 引入處理文件路徑的庫
from threading import Thread  # 引入多執行緒的庫

import numpy as np  # 引入處理數組的庫
import torch.distributed as dist  # 引入分佈式訓練庫
import torch.nn as nn  # 引入PyTorch神經網絡模組
import torch.nn.functional as F  # 引入PyTorch的神經網絡功能庫
import torch.optim as optim  # 引入PyTorch的優化器
import torch.optim.lr_scheduler as lr_scheduler  # 引入PyTorch的學習率調整器
import torch.utils.data  # 引入PyTorch的數據加載庫
import yaml  # 引入YAML配置文件解析庫
from torch.cuda import amp  # 引入PyTorch的自動混合精度訓練
from torch.nn.parallel import DistributedDataParallel as DDP  # 引入分佈式訓練庫
from torch.utils.tensorboard import SummaryWriter  # 引入TensorBoard日誌庫
from tqdm import tqdm  # 引入進度條庫

import test  # 引入測試模塊，用於獲取每個時期後的mAP
from models.experimental import attempt_load  # 引入模型初始化的函數
from models.yolo import Model  # 引入YOLO模型
from utils.autoanchor import check_anchors  # 引入檢查錨點的函數
from utils.datasets import create_dataloader  # 引入數據加載函數
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr  # 引入各種輔助函數
from utils.google_utils import attempt_download  # 引入Google驅動下載函數
from utils.loss import ComputeLoss, ComputeLossOTA  # 引入損失函數
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution  # 引入繪圖函數
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel  # 引入PyTorch輔助函數
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume  # 引入Wandb日誌記錄庫

from models.common import *

logger = logging.getLogger(__name__)

def train(hyp, opt, device, tb_writer=None):
    # 訓練函數，包括訓練超參數、選項、設備和TensorBoard記錄器作為輸入

    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))  # 記錄超參數

    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze

    # 目錄設定
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # 創建目錄
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # 保存運行設置
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # 配置
    plots = not opt.evolve  # 創建繪圖
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # 數據字典
    is_coco = opt.data.endswith('coco.yaml')

    # 記錄- 在檢查數據集之前執行這個步驟。可能會更新data_dict
    loggers = {'wandb': None}  # 記錄器字典
    if rank in [-1, 0]:
        opt.hyp = hyp  # 添加超參數
        run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(
            weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger可能會更新權重、時期（如果正在恢復訓練）

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # 類別數
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # 類別名稱
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # 檢查

    # 模型
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # 下載權重文件（如果本地找不到）
        ckpt = torch.load(weights, map_location=device)  # 載入檢查點
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 創建模型
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # 排除的鍵
        state_dict = ckpt['model'].float().state_dict()  # 轉換成FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # 取交集
        model.load_state_dict(state_dict, strict=False)  # 載入權重
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # 報告
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 創建模型
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # 檢查數據集
    train_path = data_dict['train']
    test_path = data_dict['val']

    # 凍結層
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # 要凍結的參數名稱（全局或局部）
    for k, v in model.named_parameters():
        v.requires_grad = True  # 訓練所有層
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    # 優化器
    nbs = 64  # 標稱批量大小
    accumulate = max(round(nbs / total_batch_size), 1)  # 優化之前累積損失
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # 縮放 weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # 優化器參數組
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # 偏差
        if hasattr(v, 'act_alpha'):
            pg0.append(v.act_alpha)
        if hasattr(v, 'wgt_alpha'):
            pg0.append(v.wgt_alpha)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # 無 decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # 應用 decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        # ...（一些其他優化器參數設定）

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # 調整 beta1 到 momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # 添加帶有 weight_decay 的 pg1
    optimizer.add_param_group({'params': pg2})  # 添加 pg2（偏差）
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # 調度器
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # 線性
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # 余弦 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA（Exponential Moving Average）
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # 恢復訓練
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # 恢復優化器
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # 恢復 EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # 恢復訓練結果
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # 寫入 results.txt

        # 恢復時期
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s 訓練到 %g 個時期已經完成，無法恢復。' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s 已訓練了 %g 個時期。需要進行額外的 %g 個時期微調訓練。' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # 進行微調訓練的額外時期

        del ckpt, state_dict

    # 圖像大小
    gs = max(int(model.stride.max()), 32)  # 網格大小（最大步長）
    nl = model.model[-1].nl  # 檢測層的數量（用於縮放 hyp['obj']）
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # 驗證 imgsz 是否是 gs 的倍數

    # 分佈式訓練（DDP）模式
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 同步批量標準化（SyncBatchNorm）
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('使用 SyncBatchNorm()')

    # 訓練數據加載器
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # 最大的標籤類別
    nb = len(dataloader)  # 批次數量
    assert mlc < nc, '標籤類別 %g 超出了 nc=%g 在 %s 中。可能的類別標籤是 0-%g' % (mlc, nc, opt.data, nc - 1)

    # 當處於第 0 或 -1 個進程時，執行下列代碼
    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # 測試數據加載器
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # 類別
            if plots:
                # 繪製標籤
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchor（錨點）
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # 預先減少錨點精度

    for epoch in range(start_epoch, epochs):  # 迭代每個訓練週期（epoch）

        if epoch == 295:
            # 在第295個epoch時，設定特定層的量化參數為True
            for m in model.modules():
                if isinstance(m, DSQuantConv2d) or isinstance(m, DSQuantConv2d_BNFoldVP):
                    m.quant = True

        model.train()  # 將模型設定為訓練模式

        # 更新圖像權重（可選）
        if opt.image_weights:
            # 生成索引
            if rank in [-1, 0]:
                # 計算類別權重（class weights）
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
                # 計算圖像權重（image weights）
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
                # 根據圖像權重隨機選擇索引
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)
            # 如果使用分佈式訓練，則進行廣播
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(4, device=device)  # 儲存平均損失
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # 進度條
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # 迭代每個批次

            ni = i + nb * epoch  # 已整合的批次數（從訓練開始計算）
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # 轉換圖像為浮點數格式並歸一化

            # 開始階段（warmup）
            if ni <= nw:
                xi = [0, nw]  # x 插值範圍
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # 調整學習率，初始時bias的學習率由0.1降到lr0，其他參數的學習率由0.0升到lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        # 調整動量（momentum）
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # 多尺度訓練
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # 隨機選擇尺寸
                sf = sz / max(imgs.shape[2:])  # 尺寸縮放因子
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # 新尺寸（擴展到gs的倍數）
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # 正向傳播
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # 前向傳播

                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # 損失（根據OTA算法）
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # 損失
                if rank != -1:
                    loss *= opt.world_size  # 在DDP模式下，梯度平均分佈到不同設備
                if opt.quad:
                    loss *= 4.

            # 反向傳播
            scaler.scale(loss).backward()

            # 優化
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # 顯示訓練進度
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # 更新平均損失
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # GPU記憶體使用情況
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # 顯示圖像（如果啟用並且批次數小於10）
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # 圖像文件名
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                elif plots and ni == 10 and wandb_logger.wandb:
                    # 如果使用WandB記錄器，則將部分圖像記錄到WandB
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                save_dir.glob('train*.jpg') if x.exists()]})

            # 結束訓練批次
        # 結束訓練週期

        # 設定學習率（Scheduler）
        lr = [x['lr'] for x in optimizer.param_groups]  # 用於TensorBoard
        scheduler.step()  # 調整學習率

        # DDP進程0或單GPU
        if rank in [-1, 0]:
            # 計算mAP（平均精確率）
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # 計算mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric)

            # 寫入結果
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # 附加指標和驗證損失
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # 記錄
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # 訓練損失
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # 驗證損失
                    'x/lr0', 'x/lr1', 'x/lr2']  # 參數
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # 加入TensorBoard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # WandB

            # 更新最佳mAP
            fi = fitness(np.array(results).reshape(1, -1))  # 加權組合[P、R、mAP@.5、mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # 儲存模型
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # 如果需要儲存
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # 儲存最新、最佳模型，並刪除不需要的模型
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                if epoch == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif ((epoch + 1) % 25) == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif epoch >= (epochs - 5):
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # 結束訓練過程
    # 結束訓練
    if rank in [-1, 0]:
        # 繪製圖表
        if plots:
            plot_results(save_dir=save_dir)  # 儲存為results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                          if (save_dir / f).exists()]})
        # 測試最佳模型（best.pt）
        logger.info('%g 個週期在%.3f小時內完成。\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # 如果是COCO數據集
            for m in (last, best) if best.exists() else (last):  # 速度和mAP測試
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco,
                                          v5_metric=opt.v5_metric)

        # 移除優化器（strip optimizers）
        final = best if best.exists() else last  # 最終模型
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # 去除優化器
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # 上傳模型權重
        if wandb_logger.wandb and not opt.evolve:  # 記錄去除優化器的模型
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()  # 完成W&B日誌
    else:
        dist.destroy_process_group()  # 銷毀DDP進程組
    torch.cuda.empty_cache()  # 釋放GPU內存
    return results


if __name__ == '__main__':
    # 解析命令行參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()

    # 設定DDP變數
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    #if opt.global_rank in [-1, 0]:
    #    check_git_status()
    #    check_requirements()

    # 如果是恢復訓練，檢查W&B恢復或指定的檢查點
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # 恢復中斷的訓練過程
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # 指定或最近的路徑
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # 替換參數
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # 恢復參數
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # 檢查文件
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # 延伸到2個尺寸（訓練、測試）
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # 增加保存路徑

    # DDP模式（分布式訓練）
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # 分布式後端初始化
        assert opt.batch_size % opt.world_size == 0, '--batch-size必須是CUDA設備數的倍數'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # 超參數（Hyperparameters）
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # 載入超參數設置

    # 訓練
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # 初始化記錄器
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}請運行'tensorboard --logdir {opt.project}'，在http://localhost:6006/查看")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # 超參數進化（可選）
    else:
        # 超參數進化元數據（突變比例0-1，下限，上限）
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # 初始學習率（SGD=1E-2，Adam=1E-3）
            # ...（省略部分超參數的元數據設置）...
        }
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # 加載超參數字典
            if 'anchors' not in hyp:  # 在hyp.yaml中沒有註釋錨點
                hyp['anchors'] = 3
                
        assert opt.local_rank == -1, '--evolve模式尚未實現DDP模式'
        opt.notest, opt.nosave = True, True  # 只測試/保存最終的epoch
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # 將最佳結果保存到這裡
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # 下載evolve.txt（如果存在）

        for _ in range(300):  # 進化的代數
            if Path('evolve.txt').exists():  # 如果存在evolve.txt：選擇最佳的超參數並突變
                # 選擇父代（parent）
                parent = 'single'  # 父代選擇方法：'single'或'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # 考慮的前幾個結果的數量
                x = x[np.argsort(-fitness(x))][:n]  # 前n個突變
                w = fitness(x) - fitness(x).min()  # 權重
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # 隨機選擇
                    x = x[random.choices(range(n), weights=w)[0]]  # 權重選擇
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 權重組合

                # 突變
                mp, s = 0.8, 0.2  # 突變概率、sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # 增益0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # 突變直到發生變化（防止重復）
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # 突變

            # 限制到上下限
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # 下限
                hyp[k] = min(hyp[k], v[2])  # 上限
                hyp[k] = round(hyp[k], 5)  # 有效位數

            # 訓練突變
            results = train(hyp.copy(), opt, device)

            # 寫入突變結果
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # 繪製結果
        plot_evolution(yaml_file)
        print(f'超參數進化完成。最佳結果已保存為：{yaml_file}\n'
            f'使用這些超參數訓練新模型的命令：$ python train.py --hyp {yaml_file}')
