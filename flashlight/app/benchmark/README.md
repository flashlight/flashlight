# Benchmark
This is a standard benchmark tool to measure the training performance of all Flashlight applications. It is independent from training pipeline, so that we can run it as a unit test with only one button click.

## Models

### Image classification

- [Vision Trasnformer (ViT-Base)](https://arxiv.org/abs/2010.11929)
- [ResNet-34](https://arxiv.org/abs/1512.03385)
- [ResNet-50](https://arxiv.org/abs/1512.03385)

### Object detection
- [DETR ](https://arxiv.org/abs/2005.12872)

### Language Modeling
- [Transformer (adaptive embedding + adaptive softmax)](https://arxiv.org/abs/1809.10853)

### Speech Recognition
- [Transformer (RASR)](https://arxiv.org/abs/2010.11745)

(More to come soon).


## Performance

### NVIDIA V100 GPUs

#### Throughputs
We measure the training throughputs of different models under typical training setups on different number of GPUs. The throughput is defined as `images/sec` for image classification and object detection, `tokens/sec` for language modeling and `sec/sec` for speech recognition. We assume each input frame represents 1 token for language modeling and 10ms of speech audios for recognition.

|      Model      |        Input        | Precision | GPU Mem <br>Usage (GB) | Throughput <br>1x GPU | Throughput <br>8x GPU | Throughput <br>16x GPU | Throughput<br> 32x GPU |
|:---------------:|:-------------------:|:---------:|:----------------------:|:---------------------:|:---------------------:|:----------------------:|:----------------------:|
|    ViT-Base     |  224 x 224 x 3 x 64 |    FP32   |          10.83         |         93.95         |         742.64        |         1482.57        |         2962.59        |
|                 |                     |    AMP    |          7.22          |         263.42        |        2020.85        |         3975.49        |         7931.81        |
|    ResNet-34    | 224 x 224 x 3 x 192 |    FP32   |          11.59         |         676.09        |        5298.76        |        10525.59        |        20899.64        |
|                 |                     |    AMP    |          5.71          |         941.41        |        7220.01        |        14384.46        |        28385.74        |
|    ResNet-50    | 224 x 224 x 3 x 192 |    FP32   |          28.78         |         309.43        |        2453.81        |         4887.09        |         9717.69        |
|                 |                     |    AMP    |          14.08         |         560.42        |        4333.78        |         8595.32        |        17161.24        |
|      DETR       |  800x 800 x 3 x 12  |    FP32   |          21.57         |         21.64         |         169.26        |         340.47         |         675.96         |
|                 |                     |    AMP    |          11.33         |         22.33         |         176.46        |         350.08         |         694.31         |
|  LM Transformer |   2048 x 1 x 1 x 1  |    FP32   |          17.78         |        5478.56        |        41649.23       |        82030.77        |        156341.99       |
|                 |                     |    AMP    |          14.77         |        11641.56       |        83302.7        |        144121.39       |        271310.93       |
| ASR Transformer |  1500 x 1 x 80 x 8  |    FP32   |          16.19         |         181.4         |        1407.62        |         2804.42        |         5584.33        |
|                 |                     |    AMP    |          11.76         |         487.59        |        3527.33        |         5172.56        |        13327.17        |
