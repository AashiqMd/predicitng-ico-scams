**full**
python hydra_main.py task=full model=lin num_feats=18 num_epochs=120
python hydra_main.py task=full model=linb num_feats=18 num_epochs=120 hdim=128 num_layers=1
python hydra_main.py task=full model=mlp num_feats=18 num_epochs=120 hdim=64 num_layers=1

**bow**
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=4 hdim=4
loss: 0.83385, acc: 0.588, prec: 0.488, rec: 0.601, f1: 0.539, doc_acc: 0.660, doc_prec: 0.577, doc_rec: 0.750, doc_f1: 0.652
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=4 hdim=4 dropout=0.5
loss: 0.81107, acc: 0.583, prec: 0.481, rec: 0.544, f1: 0.510, doc_acc: 0.702, doc_prec: 0.625, doc_rec: 0.750, doc_f1: 0.682
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=nltk model=bow edim=4 hdim=4 dropout=0.5
loss: 0.82171, acc: 0.564, prec: 0.462, rec: 0.542, f1: 0.499, doc_acc: 0.702, doc_prec: 0.650, doc_rec: 0.650, doc_f1: 0.650

python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=4 hdim=32
loss: 0.88198, acc: 0.580, prec: 0.480, rec: 0.592, f1: 0.530, doc_acc: 0.681, doc_prec: 0.609, doc_rec: 0.700, doc_f1: 0.651
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=4 hdim=32 dropout=0.5
loss: 0.85226, acc: 0.577, prec: 0.477, rec: 0.601, f1: 0.532, doc_acc: 0.681, doc_prec: 0.609, doc_rec: 0.700, doc_f1: 0.651

python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=8 hdim=8
loss: 0.90173, acc: 0.566, prec: 0.466, rec: 0.579, f1: 0.516, doc_acc: 0.660, doc_prec: 0.577, doc_rec: 0.750, doc_f1: 0.652
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=nltk model=bow edim=8 hdim=8
loss: 0.92636, acc: 0.565, prec: 0.465, rec: 0.580, f1: 0.516, doc_acc: 0.660, doc_prec: 0.583, doc_rec: 0.700, doc_f1: 0.636
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=8 hdim=8 dropout=0.5
loss: 0.83755, acc: 0.563, prec: 0.464, rec: 0.603, f1: 0.525, doc_acc: 0.681, doc_prec: 0.600, doc_rec: 0.750, doc_f1: 0.667
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=nltk model=bow edim=8 hdim=8 dropout=0.5
loss: 0.86099, acc: 0.560, prec: 0.462, rec: 0.615, f1: 0.528, doc_acc: 0.617, doc_prec: 0.542, doc_rec: 0.650, doc_f1: 0.591

python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=8 hdim=16
loss: 0.90970, acc: 0.571, prec: 0.469, rec: 0.564, f1: 0.512, doc_acc: 0.681, doc_prec: 0.609, doc_rec: 0.700, doc_f1: 0.651

python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=16 hdim=8
loss: 0.95037, acc: 0.572, prec: 0.471, rec: 0.571, f1: 0.516, doc_acc: 0.681, doc_prec: 0.609, doc_rec: 0.700, doc_f1: 0.651
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=16 hdim=8 num_layers=2
loss: 1.06371, acc: 0.562, prec: 0.461, rec: 0.557, f1: 0.504, doc_acc: 0.660, doc_prec: 0.591, doc_rec: 0.650, doc_f1: 0.619


python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=16 hdim=16 num_layers=2
loss: 1.05867, acc: 0.567, prec: 0.465, rec: 0.551, f1: 0.504, doc_acc: 0.681, doc_prec: 0.609, doc_rec: 0.700, doc_f1: 0.651
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=16 hdim=16 weight_decay=5e-1
loss: 0.82900, acc: 0.573, prec: 0.473, rec: 0.609, f1: 0.533, doc_acc: 0.638, doc_prec: 0.560, doc_rec: 0.700, doc_f1: 0.622

python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=16 hdim=16 nonlin=gelu
loss: 0.95754, acc: 0.568, prec: 0.468, rec: 0.579, f1: 0.518, doc_acc: 0.745, doc_prec: 0.682, doc_rec: 0.750, doc_f1: 0.714
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=16 hdim=16
loss: 0.94141, acc: 0.571, prec: 0.470, rec: 0.578, f1: 0.518, doc_acc: 0.745, doc_prec: 0.682, doc_rec: 0.750, doc_f1: 0.714
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=16 hdim=16 weight_decay=5e-2
loss: 0.94141, acc: 0.571, prec: 0.470, rec: 0.578, f1: 0.518, doc_acc: 0.745, doc_prec: 0.682, doc_rec: 0.750, doc_f1: 0.714
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=16 hdim=16 weight_decay=5e-8
loss: 0.94696, acc: 0.570, prec: 0.470, rec: 0.576, f1: 0.517, doc_acc: 0.745, doc_prec: 0.682, doc_rec: 0.750, doc_f1: 0.714
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=16 hdim=16 dropout=0.25
loss: 0.96820, acc: 0.570, prec: 0.468, rec: 0.565, f1: 0.512, doc_acc: 0.745, doc_prec: 0.682, doc_rec: 0.750, doc_f1: 0.714
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=16 hdim=16 dropout=0.5
loss: 0.90871, acc: 0.577, prec: 0.476, rec: 0.579, f1: 0.523, doc_acc: 0.745, doc_prec: 0.682, doc_rec: 0.750, doc_f1: 0.714
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=16 hdim=16 dropout=0.8
loss: 0.83836, acc: 0.572, prec: 0.472, rec: 0.595, f1: 0.527, doc_acc: 0.745, doc_prec: 0.682, doc_rec: 0.750, doc_f1: 0.714
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=nltk model=bow edim=16 hdim=16
loss: 1.02184, acc: 0.550, prec: 0.449, rec: 0.561, f1: 0.499, doc_acc: 0.681, doc_prec: 0.609, doc_rec: 0.700, doc_f1: 0.651

python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=32 hdim=32
loss: 1.07995, acc: 0.561, prec: 0.460, rec: 0.553, f1: 0.502, doc_acc: 0.660, doc_prec: 0.583, doc_rec: 0.700, doc_f1: 0.636

python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=bow edim=128 hdim=128
loss: 1.26093, acc: 0.539, prec: 0.436, rec: 0.522, f1: 0.475, doc_acc: 0.660, doc_prec: 0.591, doc_rec: 0.650, doc_f1: 0.619

python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=nltk model=bow edim=128 hdim=128
loss: 1.36046, acc: 0.527, prec: 0.427, rec: 0.530, f1: 0.473, doc_acc: 0.681, doc_prec: 0.609, doc_rec: 0.700, doc_f1: 0.651


**rnn**
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=4 hdim=4 (10 epochs)
loss: 0.81007, acc: 0.600, prec: 1.000, rec: 0.001, f1: 0.001, doc_acc: 0.596, doc_prec: 1.000, doc_rec: 0.050, doc_f1: 0.095
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=4 hdim=4 num_layers=2
loss: 0.98036, acc: 0.556, prec: 0.457, rec: 0.582, f1: 0.512, doc_acc: 0.681, doc_prec: 0.609, doc_rec: 0.700, doc_f1: 0.651


python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=8 hdim=8
loss: 1.10729, acc: 0.546, prec: 0.442, rec: 0.517, f1: 0.477, doc_acc: 0.723, doc_prec: 0.684, doc_rec: 0.650, doc_f1: 0.667
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=8 hdim=8 dropout=0.5
loss: 0.81066, acc: 0.413, prec: 0.402, rec: 0.953, f1: 0.565, doc_acc: 0.447, doc_prec: 0.435, doc_rec: 1.000, doc_f1: 0.606
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=8 hdim=8 num_layers=2
loss: 1.09816, acc: 0.547, prec: 0.445, rec: 0.535, f1: 0.486, doc_acc: 0.681, doc_prec: 0.600, doc_rec: 0.750, doc_f1: 0.667
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=8 hdim=8 num_layers=2 dropout=0.5
loss: 0.81013, acc: 0.600, prec: 0.425, rec: 0.003, f1: 0.006, doc_acc: 0.574, doc_prec: 0.000, doc_rec: 0.000, doc_f1: 0.000
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=8 hdim=8 num_layers=3
loss: 1.03847, acc: 0.552, prec: 0.450, rec: 0.533, f1: 0.488, doc_acc: 0.638, doc_prec: 0.565, doc_rec: 0.650, doc_f1: 0.605

python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=16 hdim=16
loss: 1.22562, acc: 0.535, prec: 0.433, rec: 0.522, f1: 0.473, doc_acc: 0.702, doc_prec: 0.636, doc_rec: 0.700, doc_f1: 0.667
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=16 hdim=16 dropout=0.5
loss: 0.88423, acc: 0.585, prec: 0.483, rec: 0.514, f1: 0.498, doc_acc: 0.723, doc_prec: 0.706, doc_rec: 0.600, doc_f1: 0.649
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=16 hdim=16 num_layers=2
loss: 1.20073, acc: 0.530, prec: 0.426, rec: 0.506, f1: 0.463, doc_acc: 0.660, doc_prec: 0.577, doc_rec: 0.750, doc_f1: 0.652
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=16 hdim=16 num_layers=3
loss: 1.26505, acc: 0.531, prec: 0.427, rec: 0.505, f1: 0.463, doc_acc: 0.638, doc_prec: 0.565, doc_rec: 0.650, doc_f1: 0.605

python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=32 hdim=32 rnn_bidir=True
loss: 1.48743, acc: 0.529, prec: 0.423, rec: 0.486, f1: 0.452, doc_acc: 0.681, doc_prec: 0.619, doc_rec: 0.650, doc_f1: 0.634
python hydra_main.py task=paper_sents bs=64 num_epochs=25 tokenizer=re model=rnn edim=32 hdim=32 rnn_out=output
loss: 1.50795, acc: 0.515, prec: 0.411, rec: 0.492, f1: 0.448, doc_acc: 0.660, doc_prec: 0.583, doc_rec: 0.700, doc_f1: 0.636
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=32 hdim=16 rnn_out=output rnn_bidir=True
loss: 1.29858, acc: 0.524, prec: 0.422, rec: 0.510, f1: 0.462, doc_acc: 0.681, doc_prec: 0.619, doc_rec: 0.650, doc_f1: 0.634
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=32 hdim=32
loss: 1.44243, acc: 0.513, prec: 0.411, rec: 0.498, f1: 0.450, doc_acc: 0.681, doc_prec: 0.609, doc_rec: 0.700, doc_f1: 0.651
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=32 hdim=32 rnn_type=lstm
loss: 1.18346, acc: 0.542, prec: 0.438, rec: 0.513, f1: 0.472, doc_acc: 0.681, doc_prec: 0.600, doc_rec: 0.750, doc_f1: 0.667
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=32 hdim=32 rnn_type=lstm dropout=0.5
loss: 0.99216, acc: 0.549, prec: 0.450, rec: 0.580, f1: 0.507, doc_acc: 0.617, doc_prec: 0.545, doc_rec: 0.600, doc_f1: 0.571
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=32 hdim=32 dropout=0.5
loss: 0.91800, acc: 0.580, prec: 0.479, rec: 0.567, f1: 0.519, doc_acc: 0.702, doc_prec: 0.625, doc_rec: 0.750, doc_f1: 0.682
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=32 hdim=32 dropout=0.5 rnn_dropout=True
loss: 1.02590, acc: 0.544, prec: 0.447, rec: 0.597, f1: 0.511, doc_acc: 0.681, doc_prec: 0.600, doc_rec: 0.750, doc_f1: 0.667
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=32 hdim=32 num_layers=2 dropout=0.5
loss: 0.87139, acc: 0.580, prec: 0.478, rec: 0.544, f1: 0.509, doc_acc: 0.660, doc_prec: 0.591, doc_rec: 0.650, doc_f1: 0.619

python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=64 hdim=64
loss: 1.76615, acc: 0.515, prec: 0.407, rec: 0.468, f1: 0.436, doc_acc: 0.702, doc_prec: 0.636, doc_rec: 0.700, doc_f1: 0.667
python hydra_main.py task=paper_sents bs=128 num_epochs=25 tokenizer=re model=rnn edim=64 hdim=64 dropout=0.5
loss: 0.97800, acc: 0.572, prec: 0.469, rec: 0.534, f1: 0.499, doc_acc: 0.702, doc_prec: 0.650, doc_rec: 0.650, doc_f1: 0.650

python hydra_main.py task=paper_sents bs=64 num_epochs=25 tokenizer=re model=rnn edim=128 hdim=128
loss: 3.12236, acc: 0.512, prec: 0.399, rec: 0.435, f1: 0.416, doc_acc: 0.723, doc_prec: 0.684, doc_rec: 0.650, doc_f1: 0.667
python hydra_main.py task=paper_sents bs=64 num_epochs=25 tokenizer=re model=rnn edim=128 hdim=128 dropout=0.5
loss: 1.44823, acc: 0.547, prec: 0.440, rec: 0.491, f1: 0.464, doc_acc: 0.723, doc_prec: 0.684, doc_rec: 0.650, doc_f1: 0.667