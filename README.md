# AKPFL
AKPFL: A Personalized Federated Learning  Architecture to Alleviate Statistical Heterogeneity

http://poster-openaccess.com/article_detail.php?paper_id=2521&conf=ICIC&year=2025




*If you need another data set, just write another code to download it and then use the utils.*

### Examples for **MNIST**
- MNIST
    ```
    cd ./dataset
    # python generate_MNIST.py iid - - # for iid and unbalanced scenario
    # python generate_MNIST.py iid balance - # for iid and balanced scenario
    # python generate_MNIST.py noniid - pat # for pathological noniid and unbalanced scenario
    python generate_MNIST.py noniid - dir # for practical noniid and unbalanced scenario
    # python generate_MNIST.py noniid - exdir # for Extended Dirichlet strategy 
    ```

The output of `python generate_MNIST.py noniid - dir`
```
Number of classes: 10
Client 0         Size of data: 2630      Labels:  [0 1 4 5 7 8 9]
                 Samples of labels:  [(0, 140), (1, 890), (4, 1), (5, 319), (7, 29), (8, 1067), (9, 184)]
--------------------------------------------------
Client 1         Size of data: 499       Labels:  [0 2 5 6 8 9]
                 Samples of labels:  [(0, 5), (2, 27), (5, 19), (6, 335), (8, 6), (9, 107)]
--------------------------------------------------
Client 2         Size of data: 1630      Labels:  [0 3 6 9]
                 Samples of labels:  [(0, 3), (3, 143), (6, 1461), (9, 23)]
--------------------------------------------------
```
<details>
    <summary>Show more</summary>

    Client 3         Size of data: 2541      Labels:  [0 4 7 8]
                     Samples of labels:  [(0, 155), (4, 1), (7, 2381), (8, 4)]
    --------------------------------------------------
    Client 4         Size of data: 1917      Labels:  [0 1 3 5 6 8 9]
                     Samples of labels:  [(0, 71), (1, 13), (3, 207), (5, 1129), (6, 6), (8, 40), (9, 451)]
    --------------------------------------------------
    Client 5         Size of data: 6189      Labels:  [1 3 4 8 9]
                     Samples of labels:  [(1, 38), (3, 1), (4, 39), (8, 25), (9, 6086)]
    --------------------------------------------------
    Client 6         Size of data: 1256      Labels:  [1 2 3 6 8 9]
                     Samples of labels:  [(1, 873), (2, 176), (3, 46), (6, 42), (8, 13), (9, 106)]
    --------------------------------------------------
    Client 7         Size of data: 1269      Labels:  [1 2 3 5 7 8]
                     Samples of labels:  [(1, 21), (2, 5), (3, 11), (5, 787), (7, 4), (8, 441)]
    --------------------------------------------------
    Client 8         Size of data: 3600      Labels:  [0 1]
                     Samples of labels:  [(0, 1), (1, 3599)]
    --------------------------------------------------
    Client 9         Size of data: 4006      Labels:  [0 1 2 4 6]
                     Samples of labels:  [(0, 633), (1, 1997), (2, 89), (4, 519), (6, 768)]
    --------------------------------------------------
    Client 10        Size of data: 3116      Labels:  [0 1 2 3 4 5]
                     Samples of labels:  [(0, 920), (1, 2), (2, 1450), (3, 513), (4, 134), (5, 97)]
    --------------------------------------------------
    Client 11        Size of data: 3772      Labels:  [2 3 5]
                     Samples of labels:  [(2, 159), (3, 3055), (5, 558)]
    --------------------------------------------------
    Client 12        Size of data: 3613      Labels:  [0 1 2 5]
                     Samples of labels:  [(0, 8), (1, 180), (2, 3277), (5, 148)]
    --------------------------------------------------
    Client 13        Size of data: 2134      Labels:  [1 2 4 5 7]
                     Samples of labels:  [(1, 237), (2, 343), (4, 6), (5, 453), (7, 1095)]
    --------------------------------------------------
    Client 14        Size of data: 5730      Labels:  [5 7]
                     Samples of labels:  [(5, 2719), (7, 3011)]
    --------------------------------------------------
    Client 15        Size of data: 5448      Labels:  [0 3 5 6 7 8]
                     Samples of labels:  [(0, 31), (3, 1785), (5, 16), (6, 4), (7, 756), (8, 2856)]
    --------------------------------------------------
    Client 16        Size of data: 3628      Labels:  [0]
                     Samples of labels:  [(0, 3628)]
    --------------------------------------------------
    Client 17        Size of data: 5653      Labels:  [1 2 3 4 5 7 8]
                     Samples of labels:  [(1, 26), (2, 1463), (3, 1379), (4, 335), (5, 60), (7, 17), (8, 2373)]
    --------------------------------------------------
    Client 18        Size of data: 5266      Labels:  [0 5 6]
                     Samples of labels:  [(0, 998), (5, 8), (6, 4260)]
    --------------------------------------------------
    Client 19        Size of data: 6103      Labels:  [0 1 2 3 4 9]
                     Samples of labels:  [(0, 310), (1, 1), (2, 1), (3, 1), (4, 5789), (9, 1)]
    --------------------------------------------------
    Total number of samples: 70000
    The number of train samples: [1972, 374, 1222, 1905, 1437, 4641, 942, 951, 2700, 3004, 2337, 2829, 2709, 1600, 4297, 4086, 2721, 4239, 3949, 4577]
    The number of test samples: [658, 125, 408, 636, 480, 1548, 314, 318, 900, 1002, 779, 943, 904, 534, 1433, 1362, 907, 1414, 1317, 1526]

    Saving to disk.

    Finish generating dataset.
</details>

## Models
- for MNIST and Fashion-MNIST

    1. Mclr_Logistic(1\*28\*28)
    2. LeNet()
    3. DNN(1\*28\*28, 100) # non-convex

- for Cifar10, Cifar100 and Tiny-ImageNet

    1. Mclr_Logistic(3\*32\*32)
    2. FedAvgCNN()
    3. DNN(3\*32\*32, 100) # non-convex
    4. ResNet18, AlexNet, MobileNet, GoogleNet, etc.

- for AG_News and Sogou_News

    1. LSTM()
    2. fastText() in [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/) 
    3. TextCNN() in [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/)
    4. TransformerModel() in [Attention is all you need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

- for AmazonReview

    1. AmazonMLP() in [Curriculum manager for source selection in multi-source domain adaptation](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_36)

- for Omniglot

    1. FedAvgCNN()

- for HAR and PAMAP

    1. HARCNN() in [Convolutional neural networks for human activity recognition using mobile sensors](https://eudl.eu/pdf/10.4108/icst.mobicase.2014.257786)

## Environments
Install [CUDA](https://developer.nvidia.com/cuda-11-6-0-download-archive). 

Install [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) and activate conda. 

```bash
conda env create -f env_cuda_latest.yaml # You may need to downgrade the torch using pip to match the CUDA version
```

## How to start simulating (examples for FedAvg)

- Create proper environments (see [Environments](#environments)).

- Download [this project](https://github.com/TsingZ0/PFLlib) to an appropriate location using [git](https://git-scm.com/).
    ```bash
    git clone https://github.com/TsingZ0/PFLlib.git
    ```

- Build evaluation scenarios (see [Datasets and scenarios (updating)](#datasets-and-scenarios-updating)).

- Run evaluation: 
    ```bash
    cd ./system
    python main.py -data MNIST -m cnn -algo FedAvg -gr 2000 -did 0 # using the MNIST dataset, the FedAvg algorithm, and the 4-layer CNN model
    ```

**Note**: It is preferable to tune algorithm-specific hyper-parameters before using any algorithm on a new machine. 
