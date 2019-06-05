# TCN-Analysis
This repository aims at studying some characteristics of Temporal Neural Network (TCN). \
Original code: https://github.com/locuslab/TCN


## Sequential MNIST dataset. 
MNIST dataset has 60k training and 10k testing samples. Each is a 28x28 image (784 pixels). 
TCN model observe all 784 pixels in a sequence manner. In each time step, it obseves the next pixel
as inputs and produce the prediction result. 

In classification task, the result at the last time step will be used as input to a linear layer 
for classification. Thus, it is important to set kernel_size, #levels, #dilation, so that the receptive field
covers all input pixels. In other words, the information from all pixels are learned in the last output node. 

## Replicate the results:
In the directory mnist_pixels, type
```
>> python pmnist_test.py 
```
Results after 20 training epochs: 
```
Test set: Average loss: 0.0385, Accuracy: 9891/10000 (98%)
```

## Network Architectures
<details>
<summary> Click to see network model for sequential MNIST </summary> <p>

```
Namespace(batch_size=64, clip=-1, cuda=True, dropout=0.05, epochs=20, ksize=7, levels=8, log_interval=100, lr=0.002, nhid=25, optim='Adam', permute=False, seed=1111)
TCN(
  (tcn): TemporalConvNet(
    (network): Sequential(
      (0): TemporalBlock(
        (conv1): Conv1d(1, 25, kernel_size=(7,), stride=(1,), padding=(6,))
        (chomp1): Chomp1d()
        (relu1): ReLU()
        (dropout1): Dropout(p=0.05)
        (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(6,))
        (chomp2): Chomp1d()
        (relu2): ReLU()
        (dropout2): Dropout(p=0.05)
        (net): Sequential(
          (0): Conv1d(1, 25, kernel_size=(7,), stride=(1,), padding=(6,))
          (1): Chomp1d()
          (2): ReLU()
          (3): Dropout(p=0.05)
          (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(6,))
          (5): Chomp1d()
          (6): ReLU()
          (7): Dropout(p=0.05)
        )
        (downsample): Conv1d(1, 25, kernel_size=(1,), stride=(1,))
        (relu): ReLU()
      )
      (1): TemporalBlock(
        (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(12,), dilation=(2,))
        (chomp1): Chomp1d()
        (relu1): ReLU()
        (dropout1): Dropout(p=0.05)
        (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(12,), dilation=(2,))
        (chomp2): Chomp1d()
        (relu2): ReLU()
        (dropout2): Dropout(p=0.05)
        (net): Sequential(
          (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(12,), dilation=(2,))
          (1): Chomp1d()
          (2): ReLU()
          (3): Dropout(p=0.05)
          (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(12,), dilation=(2,))
          (5): Chomp1d()
          (6): ReLU()
          (7): Dropout(p=0.05)
        )
        (relu): ReLU()
      )
      (2): TemporalBlock(
        (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(24,), dilation=(4,))
        (chomp1): Chomp1d()
        (relu1): ReLU()
        (dropout1): Dropout(p=0.05)
        (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(24,), dilation=(4,))
        (chomp2): Chomp1d()
        (relu2): ReLU()
        (dropout2): Dropout(p=0.05)
        (net): Sequential(
          (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(24,), dilation=(4,))
          (1): Chomp1d()
          (2): ReLU()
          (3): Dropout(p=0.05)
          (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(24,), dilation=(4,))
          (5): Chomp1d()
          (6): ReLU()
          (7): Dropout(p=0.05)
        )
        (relu): ReLU()
      )
      (3): TemporalBlock(
        (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(48,), dilation=(8,))
        (chomp1): Chomp1d()
        (relu1): ReLU()
        (dropout1): Dropout(p=0.05)
        (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(48,), dilation=(8,))
        (chomp2): Chomp1d()
        (relu2): ReLU()
        (dropout2): Dropout(p=0.05)
        (net): Sequential(
          (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(48,), dilation=(8,))
          (1): Chomp1d()
          (2): ReLU()
          (3): Dropout(p=0.05)
          (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(48,), dilation=(8,))
          (5): Chomp1d()
          (6): ReLU()
          (7): Dropout(p=0.05)
        )
        (relu): ReLU()
      )
      (4): TemporalBlock(
        (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(96,), dilation=(16,))
        (chomp1): Chomp1d()
        (relu1): ReLU()
        (dropout1): Dropout(p=0.05)
        (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(96,), dilation=(16,))
        (chomp2): Chomp1d()
        (relu2): ReLU()
        (dropout2): Dropout(p=0.05)
        (net): Sequential(
          (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(96,), dilation=(16,))
          (1): Chomp1d()
          (2): ReLU()
          (3): Dropout(p=0.05)
          (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(96,), dilation=(16,))
          (5): Chomp1d()
          (6): ReLU()
          (7): Dropout(p=0.05)
        )
        (relu): ReLU()
      )
      (5): TemporalBlock(
        (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(192,), dilation=(32,))
        (chomp1): Chomp1d()
        (relu1): ReLU()
        (dropout1): Dropout(p=0.05)
        (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(192,), dilation=(32,))
        (chomp2): Chomp1d()
        (relu2): ReLU()
        (dropout2): Dropout(p=0.05)
        (net): Sequential(
          (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(192,), dilation=(32,))
          (1): Chomp1d()
          (2): ReLU()
          (3): Dropout(p=0.05)
          (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(192,), dilation=(32,))
          (5): Chomp1d()
          (6): ReLU()
          (7): Dropout(p=0.05)
        )
        (relu): ReLU()
      )
      (6): TemporalBlock(
        (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(384,), dilation=(64,))
        (chomp1): Chomp1d()
        (relu1): ReLU()
        (dropout1): Dropout(p=0.05)
        (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(384,), dilation=(64,))
        (chomp2): Chomp1d()
        (relu2): ReLU()
        (dropout2): Dropout(p=0.05)
        (net): Sequential(
          (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(384,), dilation=(64,))
          (1): Chomp1d()
          (2): ReLU()
          (3): Dropout(p=0.05)
          (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(384,), dilation=(64,))
          (5): Chomp1d()
          (6): ReLU()
          (7): Dropout(p=0.05)
        )
        (relu): ReLU()
      )
      (7): TemporalBlock(
        (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(768,), dilation=(128,))
        (chomp1): Chomp1d()
        (relu1): ReLU()
        (dropout1): Dropout(p=0.05)
        (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(768,), dilation=(128,))
        (chomp2): Chomp1d()
        (relu2): ReLU()
        (dropout2): Dropout(p=0.05)
        (net): Sequential(
          (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(768,), dilation=(128,))
          (1): Chomp1d()
          (2): ReLU()
          (3): Dropout(p=0.05)
          (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(768,), dilation=(128,))
          (5): Chomp1d()
          (6): ReLU()
          (7): Dropout(p=0.05)
        )
        (relu): ReLU()
      )
    )
  )
  (linear): Linear(in_features=25, out_features=10, bias=True)
)
```
</p></details>

<img src="https://user-images.githubusercontent.com/13492723/58992907-46e2c400-87a9-11e9-8e5a-dc8e0d8408fc.JPG" width="500">
The network used for sequential MNIST datasets is simmilar to the one presented in orginal paper [1] (figure above). It has 
8 levels, each level is a Temporal Residual Block. Each Temporal Residul Block consists of 2 convolution layers with same kernel size, same dilation. It just one on top of the other. The simmilar image presetned in original paper below: 
<img src="https://user-images.githubusercontent.com/13492723/58993662-6a0e7300-87ab-11e9-9353-35d4f25fee89.JPG" width="500">

#### Residual Blocks: 

This followed the same idea presented in ResNet that it is better to learn the modification (residual) F(x) of input x instead of entire transformation, so output at each resdiual block is o = activate(x + F(x)). 

In TCN, the number of channels of input x and output o could be different. Look at the first resdiual block, where #in_channels = 1 
while #out_channels = 25. To have idential channels, conv1d is used in code: 
```
self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
```


## Calculating receptive field.
It is important to make sure the network's receptive field is greater than input length. Assume the kernel size is fixed (=3), the dilation size is double every layer d = 2^(l-1), where l = {1,...,L} is the index of each layer. l = 0 is input layer. We 
can calculate the number of layer L needed to cover entire input length. 

In each layer l, the history (covered) by an node is F(l) = F(l-1) + (k-1)d = F(l-1) + (k-1)2^{l-1}
= (k-1)(2^0 + ...+ 2^{l-1}) = (k-1)(2^l -1)  
Check:  
F(1) = (3-1)(2^1 - 1) = 2 (correct)  
F(2) = (3-1)(2^2 - 1) = 6 (correct)  
In our example, length of input is 784, k =7 thus the number of layer needed is  
784 = (7-1)(2^l-1) => l = 7.04. Thus, l must be at least 8 levels. 
In fact, if l = 8, the network's receptive field is 6(2^8-1) = 1530 in history size


## References. 
[1] Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling." arXiv preprint arXiv:1803.01271 (2018).
