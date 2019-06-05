# TCN-Analysis
This repository aims at studying some characteristics of Temporal Neural Network (TCN). \
Original code: https://github.com/locuslab/TCN


# Understanding sequential MNIST dataset. 


# Replicate the results:
In the directory mnist_pixels, type
```
>> python pmnist_test.py 
```
Results after 20 training epochs: 
```
Test set: Average loss: 0.0385, Accuracy: 9891/10000 (98%)
```

## Network Architectures
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
