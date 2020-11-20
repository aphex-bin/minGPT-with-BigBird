# minGPT-with-BigBird
DD2412 project at KTH by Leo Hiselius, Jonas Thunberg and Alfons Heintz, {leohi, jonthu, alfonsh}"at"kth.se

This project strives to combine two models:
  - [The minGPT model, a light weight implementation of iGPT by Andrej Karpathy](https://github.com/karpathy/minGPT) published under the MIT license
  - [The BigBird attention masking developed by Zaheer et. al](https://arxiv.org/abs/2007.14062)

# Instructions to recreate results presented in project report
- Run train.py to train GPT models.
- Rune finetune.py to finetune the model heads to a classification task.
- Run linearprobe.py to train one linear probe model per layer of each model.
- Run accuracy to check the accuracy of all of the different models.
- Run generate to generate example pictures from the full vanilla and BigBird models.

# Notes for devs

[Maybe useful example of autograd on sparse matrices in comments](https://discuss.pytorch.org/t/manually-calculate-the-gradient-of-a-sparse-matrix/86203/3)

[Video on BigBird, timestamp on block/roll implementation of sparse attention](https://youtu.be/WVPE62Gk3EM?t=1553)
