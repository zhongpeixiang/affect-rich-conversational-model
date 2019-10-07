The PyTorch code for paper: An Affect-Rich Neural Conversational Model with Biased Attention and Weighted Cross-Entropy Loss

The model is largely based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)(v0.1) and PyTorch 0.4.

## Steps

- Download data from OpenSubtitles, Cornell Movie Dialog Corpus, DailyDialog datasets or your own datasets. The data consists of parallel source (src) and target (tgt) data containing one sentence per line with tokens separated by a space.
- Download pretrained GloVe embeddings or initialize your own GloVe embeddings (e.g., 1024 dimensional embedding) using the training dataset based on https://github.com/stanfordnlp/GloVe.
- Data preprocessing: see http://opennmt.net/OpenNMT-py/options/preprocess.html.
- Model training: see http://opennmt.net/OpenNMT-py/options/train.html.
- Model evaluation: see http://opennmt.net/OpenNMT-py/options/translate.html.

## Citing
If you find this repo useful, please cite
```
@inproceedings{zhong2019affect,
  title={An Affect-Rich Neural Conversational Model with Biased Attention and Weighted Cross-Entropy Loss},
  author={Zhong, Peixiang and Wang, Di and Miao, Chunyan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={7492--7500},
  year={2019}
}
```