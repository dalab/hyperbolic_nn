# [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112)
### Python source code

Other related material can be accessed [here](http://people.inf.ethz.ch/ganeao/).


1. Prerequisites:
```
python3.5, Tensorflow 1.8, numpy, pickle, logging
```

2. Generate the 3d MLR figure from our paper.
```
python3.5 viz_mlr.py
```

3. Run the code to reproduce results from Table 1. Example of command that runs hyperbolic GRUs + one hyperbolic fully connected layer + hyperbolic MLR to embed each pair of sentences from the PREFIX10 dataset (assuming the location of this dataset is in the same directory as the source code):
```
CUDA_VISIBLE_DEVICES='' python3.5 hyp_rnn.py --base_name='' --dataset='PRFX10' --inputs_geom='hyp' --word_dim=5 --word_init_avg_norm=0.001   --cell_type='gru' --cell_non_lin='id'  --sent_geom='hyp' --bias_geom='hyp' --ffnn_geom='hyp' --ffnn_non_lin='id' --additional_features='dsq'  --dropout=1.0 --before_mlr_dim=5 --mlr_geom='hyp'  --reg_beta=0.0  --hyp_opt='rsgd' --lr_ffnn=0.01 --lr_words=0.1 --burnin='n' --proj_eps=1e-5 --batch_size=64 --root_path=./
```

The data needed in this code lives in the *_dataset folders and was generated as follows:

- SNLI data was put in a binary format using the file `binarize_snli_dataset.py` and the original [SNLI dataset](https://nlp.stanford.edu/projects/snli/)

- the PREFIX dataset was generated using the file `prefix_dataset.py`



## References
If you find this code useful for your research, please cite the following paper in your publication:
```
@article{ganea2018hyperbolic,
  title={Hyperbolic Neural Networks},
  author={Ganea, Octavian-Eugen and B{\'e}cigneul, Gary and Hofmann, Thomas},
  journal={arXiv preprint arXiv:1805.09112},
  year={2018}
}
```
