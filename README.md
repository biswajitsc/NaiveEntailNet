# EntailNet
Tensorflow implementation of neural networks for the entailment problem in natural language understanding.
A naive model using LSTMs on the SEMEVAL SICK dataset from http://alt.qcri.org/semeval2014/task1/

To run:
* First generate the google wordvector file by running GenerateVectors.c passing the
input binary file name as a parameter. We used google news wordvectors.
* Once wordvectors are generated, simply run main.py.

Some Issues:
* Confusing layer creations. I could have created a function for adding a layer
or could have used tfslim.
* The tensorlayer for the final prediction causes the model to overfit. Use a simple
2 layer MLP instead of that.
* Untuned hyperparameters.
