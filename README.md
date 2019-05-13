# SphereGAN-Tensorflow
Simple Tensorflow implementation of SphereGAN (CVPR 2019 Oral) | [paper](http://cau.ac.kr/~jskwon/paper/SphereGAN_CVPR2019.pdf) | [supplementary materials](http://cau.ac.kr/~jskwon/paper/SphereGAN_CVPR2019_SUPP.pdf)

<div align="center">
  <img src="./assets/architecture.png" height = '300px'>
</div>

## Usage
* `mnist` and `cifar10` are used inside keras
* For `your dataset`, put images like this:
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── xxx.jpg (name, format doesn't matter)
       ├── yyy.png
       └── ...
```

### Train
```
> python main.py --dataset mnist --gan_type sphere --phase train
```

### Random test
```
> python main.py --dataset mnist --gan_type sphere --phase test
```

## Analysis
###  Inverse of stereographic projection
![isp](./assets/isp_alg.png)

### Moment mode
<img src="./assets/moment.png" height = '500px'>

## Results
### Score
![score](./assets/result.png)

### Image (64x64)
* The paper only posted the results for `32x32 images`, but I also tried `64x64`
#### Mnist
![mnist](./assets/mnist_64.png)

#### Cifar10
![cifar](./assets/cifar10_64.png)

#### Lsun bedroom
![lsun](./assets/lsun_64.png)

#### CelebA
![celeb](./assets/celebA_64.png)

## Author
Junho Kim
