# Conformal CANN detector

This repository contains the code to reproduce our experiments with the *conformal CANN detector*, our method for detecting adversarial examples based on *conformal prediction* [1] and *correspondence analysis* [2].

## Getting the data
Our experiments were designed for the MNIST, Fashion-MNIST, CIFAR-10 and SVHN data sets. All of these are included in the [TensorFlow Keras](https://www.keras.io/) framework upon which our code is built, except for SVHN. You can download this data set from [this URL](http://ufldl.stanford.edu/housenumbers/). Note that we used 32x32 cropped digits (format 2).

## Running the code

To run the code, you will need to make sure all dependencies are properly installed. There is a `requirements.txt` file provided to facilitate this, assuming you have Python 3.6.9 or higher:

    pip install -r requirements.txt

There are two main scripts provided for running our experiments: `evaluate.py` and `attack.py`. The former will train baseline models and CANNs from scratch and evaluate them against the $L_\infty$ projected gradient descent attack [3]. The latter script runs our adaptive adversarial attack which specifically attempts to fool the combined detector and classifier simultaneously. Note that you must run the evaluation script before you can run the adaptive attack, since our attack requires certain information (the tuned detection thresholds) provided by `evaluate.py`.

### The evaluation script

An example command to run the evaluation is shown below:

    python evaluate.py mnist ResNet50 --eps .3

This will train a ResNet50 and CANN model for the MNIST data set and evaluate it against the PGD attack up to a relative perturbation budget of 30%. When this script is completed, it will generate a `results.json` file in the `results` folder that looks like this:

    {
        "baseline_acc": "0.99292517",
        "center_score": "0.07354409396427587",
        "thresholds": [
            "0.07523809523809524",
            "0.10634920634920635",
            "0.10793650793650794",
            "0.10634920634920635",
            "0.10634920634920635",
            "0.1061904761904762",
            "0.10634920634920635",
            "0.10714285714285714",
            "0.10714285714285714",
            "0.1061904761904762"
        ]
    }

This means the ResNet50 baseline model achieved a clean test accuracy of approximately 99.29%. The `center_score` is the mean deviation value as specified in the paper, which is about 0.074 in this case. The `thresholds` array specifies the non-conformity threshold for each perturbation budget. This JSON file contains important information that is necessary for the adaptive attack to work. The evaluation script will also produce a PDF file `curve_LinfProjectedGradientDescentAttack.pdf` which plots the robust accuracy and rejection rates as a function of the relative perturbation budget.

### The adaptive attack

When `evaluate.py` has finished, you can run the adaptive adversarial attack script `attack.py` as follows:

    python attack.py mnist ResNet50 --eps .3

This will run the adaptive attack againt the pre-trained ResNet50 and CANN models for the MNIST data set up to an $L_\infty$ relative perturbation of 30%. To do this, the attack must reuse the `center_score` and `thresholds` values produced by `evaluate.py`. It will then produce a PDF file `curve_adaptive.pdf` which plots the robust accuracy and rejection rates as a function of the relative perturbation budget.

### Shell scripts
To reproduce our results exactly, you can also run the enclosed bash shell scripts `mnist.sh`, `fashion.sh`, `cifar10.sh` and `svhn.sh` respectively for the MNIST, Fashion-MNIST, CIFAR-10 and SVHN data sets. The script `all.sh` will run all of these in sequence.

### Rolling your own
You can run our experiments against your own models and data sets by supplying a model specification, a CANN architecture and a data provider. To do this, follow these steps:

1. Create a Python script under `./datasets/dataset_name.py` which implements the `load_data`, `create_cann` and `train_cann` functions. You can probably copy the `train_cann` function as-is from existing code unless you need special procedures to fit your CANN.
2. Create a Python script under `./models/model_name.py` which implements the `create_model` and `train_baseline` functions.
3. You can now run `evaluate.py` and `attack.py` on your own data set and models.

## References

1. Shafer, G., & Vovk, V. (2008). A tutorial on conformal prediction. Journal of Machine Learning Research, 9(Mar), 371-421. [PDF](http://www.jmlr.org/papers/volume9/shafer08a/shafer08a.pdf)
2. Hsu, H., Salamatian, S., & Calmon, F. P. (2019). Correspondence analysis using neural networks. arXiv preprint arXiv:1902.07828. [PDF](https://arxiv.org/pdf/1902.07828)
3. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083. [PDF](https://arxiv.org/pdf/1706.06083)
