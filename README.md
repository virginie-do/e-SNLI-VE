# e-SNLI-VE-2.0

This repository contains the dataset and code for our paper: 

* e-SNLI-VE-2.0: Corrected Visual-Textual Entailment with Natural Language Explanations [[arXiv]](http://arxiv.org/abs/2004.03744) [1]

It will be presented at the 2020 CVPR workshop on [Fair, Data Efficient and Trusted Computer Vision](https://sites.google.com/view/fair-data-efficient-trusted-cv/home).

## Dataset

The e-SNLI-VE-2.0 dataset is located in the folder `data/`. It extends both [SNLI-VE](https://github.com/necla-ml/SNLI-VE) [2] and [e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI) [3]

Due to the Github size restrictions, the training set is split in two files, please simply merge them.

![Example from e-SNLI-VE.2.0](https://github.com/virginie-do/e-SNLI-VE/raw/master/e-snli-ve-dog-example.jpg)

## Code

### Prerequisites
1. Python 3.7 / Tensorflow 1.14
2. Flickr30k ResNet-101 / Fast R-CNN [feature files](https://drive.google.com/file/d/1-Jq5FFByurew-QvwTMz59Llg-2j00xTv/view?usp=sharing) and [image_names]
(https://drive.google.com/file/d/0B40JtotizQfxMG81TVoteHlKdFU/view) 
3. [Glove embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip)

See this [repository](https://github.com/claudiogreco/coling18-gte) for more details.

### Training and testing
~~~
python train_explain.py --train_filename='./data/e_vsnli_train.tsv'  --dev_filename='./data/e_vsnli_dev.tsv' --vectors_filename="./data/glove.840B.300d.txt" --img_names_filename='./data/image_features/flickr30k_resnet101_bottom_up_img_names.json' --img_features_filename='./data/image_features/flickr30k_resnet101_bottom_up_img_features.npy' --model_save_filename='./models/e_vsnli' --batch_size=100 --max_vocab=5000 --alpha=0.8 --buffer_size=300000

python eval_explain.py --test_filename='./data/e_vsnli_test.tsv' --model_filename='./models/e_vsnli' --img_names_filename='./data/image_features/flickr30k_resnet101_bottom_up_img_names.json' --img_features_filename='./data/image_features/flickr30k_resnet101_bottom_up_img_features.npy' --result_filename="./models/result_e_vsnli"
~~~

## Bibtex

If you use this dataset in your work, please cite our paper:

```
@misc{do2020esnlive20,
    title={e-SNLI-VE-2.0: Corrected Visual-Textual Entailment with Natural Language Explanations},
    author={Virginie Do and Oana-Maria Camburu and Zeynep Akata and Thomas Lukasiewicz},
    year={2020},
    eprint={2004.03744},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


## References

[1] Do, V., Camburu, O., Akata, Z., & Lukasiewicz, T. (2020). e-SNLI-VE-2.0: Corrected Visual-Textual Entailment with Natural Language Explanations. arXiv preprint arXiv:2004.03744.

[2] Camburu, O. M., Rocktäschel, T., Lukasiewicz, T., & Blunsom, P. (2018). e-SNLI: Natural Language Inference with Natural Language Explanations. In Advances in Neural Information Processing Systems (pp. 9539-9549).

[3] Xie, N., Lai, F., Doran, D., & Kadav, A. (2019). Visual Entailment: A Novel Task for Fine-Grained Image Understanding. arXiv preprint arXiv:1901.06706.


