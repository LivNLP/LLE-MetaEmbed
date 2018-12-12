# LLE-MetaEmbed
## Locally Linear Meta Embedding Learning.

This package implements the locally linear word meta embedding learning methods proposed in the following paper.
(please cite it if you use the code in your work)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@inproceedings{Bollegala:IJCAI:2018,
   author = {Danushka Bollegala and Koheu Hayashi and Ken-ichi Kawarabayashi},
    title = {Think Globally, Embed Locally --- Locally Linear Meta-embedding of Words},
  booktitle = {Proc. of IJCAI-ECAI},
  pages = {3970--3976},
  year = {2018}
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Usage

* Save your pre-trained source word embeddings to the "sources" directory. You could download several pre-trained word embeddings from the following links.
    * [300 dimensional GloVe embeddings](http://nlp.stanford.edu/data/glove.42B.300d.zip)
    * [300 dimensional continous bag-of-words (CBOW) embeddings](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
    * [100 dimensional Hierarchical Log bi-linar word embeddings](https://www.dropbox.com/s/f3kmthi7jttwyib/HLBL%2B100?dl=0)
    * [50 dimensional word embeddings created by Huang et al. [2012]](https://www.dropbox.com/s/owryrlbmmwjcj1h/Huang%2B50?dl=0)
    * [200 dimensional word embeddings created by Collobert and Weston [2008]](https://www.dropbox.com/s/zgakc14fl56w7l5/CW%2B200?dl=0)

* Edit ./src/sources.py and specify the paths for those source word embeddings and their dimensionalities. The vocabulary of the words for which meat embeddings will be created is in ./work/selected-words. You can add your own vocabulary by editing this file but make sure that the word emebddings for those words are available in your pre-trained source embeddings. Otherwise, we would assume that word to be missing in the source word embedding.

* Run 
    ```
        python meta-embed.py --nns [neighbourhood size] --comps [dimensionality of the meta-embeddings]
    ```
    The output meta embeddings will be written to ./work/meta-embeds