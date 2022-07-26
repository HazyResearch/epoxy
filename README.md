# Epoxy: Interactive Model Iteration with Weak Supervision and Pre-Trained Embeddings

**UPDATE 07/26/22**: Our newest code is now [Liger repo](https://github.com/HazyResearch/liger)

**UPDATE 07/20/20**: Code now supports using FAISS for fast and scalable NN
search! Tutorial example updated to use FAISS now.

Epoxy uses weak supervision and pre-trained embeddings to create models that
can train at programmatically-interactive speeds (less than 1/2 second), but
that can retain the performance of training deep networks.
This repository presents a simple proof-of-concept implementation for Epoxy
(our implementation is around 100 LOC, including docstrings).

<div>
  <img src="figs/figure_1_png.png" width="800">
</div>

In weak supervision, users write noisy _labeling functions_ that generate labels
for the data.
Historically, we have observed that these labeling functions are often
high accuracy but low coverage (each labeling function only votes on a subset of
points).
The only ways to make up the gap in the past have been to write more labeling
functions (which can get difficult as you start dealing with the long tail),
or use the labeling functions to train an end model (see, e.g.,
[FlyingSquid](https://github.com/HazyResearch/flyingsquid) for more details).

In Epoxy, we use pre-trained embeddings to get some of the benefits of training
an end model--without having to train one.
We use the embeddings to create _extended labeling functions_ through
nearest-neighbors search (improving coverage), and then use FlyingSquid to
aggregate the extended labeling functions.
This helps get some of the benefits of training a deep network, but at a
fraction of the cost.
And if you do have time to train a deep network, Epoxy can be used to generate
labels to train a downstream end model as well.

Check out our paper on [arXiv](https://arxiv.org/abs/2006.15168) for more details!

## Getting Started
* [Install](#installation) Epoxy
* Check out the [example tutorial](examples/example_tutorial.ipynb) for a simple
Jupyter notebook showing the proof of concept in this repo.

## Installation

This repository depends on FlyingSquid.
We recommend using `conda` to install FlyingSquid, and then you can install
Epoxy:

```
git clone https://github.com/HazyResearch/flyingsquid.git

cd flyingsquid

conda env create -f environment.yml
conda activate flyingsquid

pip install -e .

cd ..

git clone https://github.com/HazyResearch/epoxy.git

cd epoxy

# if you are on a machine with a GPU
pip install faiss-gpu
# if you are on a machine without a GPU
pip install faiss-cpu

pip install -e .
```

Alternatively, you can install FlyingSquid (and its dependencies) yourself,
see the [FlyingSquid repo](https://github.com/HazyResearch/flyingsquid)
for more details.


## Citation

If you use our work or found it useful, please cite our [arXiv paper](https://arxiv.org/abs/2006.15168) for now:
```
@article{chen2020train,
  author = {Mayee F. Chen and Daniel Y. Fu and Frederic Sala and Sen Wu and Ravi Teja Mullapudi and Fait Poms and Kayvon Fatahalian and Christopher R\'e},
  title = {Train and You'll Miss It: Interactive Model Iteration with Weak Supervision and Pre-Trained Embeddings},
  journal = {arXiv preprint arXiv:2006.15168},
  year = {2020},
}
```
