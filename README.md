# bill_seg
基于unet实现的票据语义分割
## 环境(Requirements)
```tensorboard (1.10.0)```

## 模型(model)
```链接: https://pan.baidu.com/s/1wGfC8XCfqA5d6BECDt6TqA 提取码: jewv 复制这段内容后打开百度网盘手机App，操作更方便哦``

## 例子🌰(Demo)

```python model.py```

## 训练(train)
- 修改train.py 中checkpoint_path 为模型路径
- 修改dataf.py 中training_data_path 为训练数据路径

```python train.py```

## 可视化实例
### 例子🌰1
![raw](https://github.com/tommyMessi/tableImageParser_tx/blob/master/tx_infer_data/vanke_2016_1241_nb_3.jpg)
![nrow](https://github.com/tommyMessi/tableImageParser_tx/blob/master/tx_infer_data/nrow/vanke_2016_1241_nb_3.jpg)
![ncol](https://github.com/tommyMessi/tableImageParser_tx/blob/master/tx_infer_data/ncol/vanke_2016_1241_nb_3.jpg)
### 例子🌰2
![raw](https://github.com/tommyMessi/tableImageParser_tx/blob/master/tx_infer_data/1.jpg)
![row](https://github.com/tommyMessi/tableImageParser_tx/blob/master/tx_infer_data/row/1.jpg)
![row](https://github.com/tommyMessi/tableImageParser_tx/tree/master/tx_infer_data/col)

## 其他
更多分享与该工程问题 关注微信公众账号 hulugeAI 进群讨论 
