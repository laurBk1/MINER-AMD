# gpu_miner_btcw_windows
CUDA GPU miner for BTCW on native Windows

You can use the same wallet.dat for all GPUs. You must tell the GPU to use a different stage2 selection number to ensure unique stage2 hashing when using the same stage1 keys.

GPU miner works on Windows10 and Windows11.


Start Bitcoin-Pow-QT
```
bitcoin-pow-qt.exe -emergencymining=1
```

Have at least 1 utxo in your wallet. 
Go to Window->Console  and  type the following command and hit the enter button on keyboard
```
generate
```


DO NOT start the BTCW GPU miner until the BTCW node has started its mining process.

# Make sure the kernel.ptx file is in the same folder as the exe
start the gpu miner for gpu number 1
```
BTCW_CUDA_MINER.exe 1
```

start the gpu miner for gpu number 2
```
BTCW_CUDA_MINER.exe 2
```

start the gpu miner for gpu number 3
```
BTCW_CUDA_MINER.exe 3
```  

Here is a typical output when mining
```
Bitcoin-PoW GPU Miner v26.5.4

Hash found - NONCE: 0707070707070707

Hash no sig low64: 6bfb0091a42a80d9

CONNECTED TO BTCW NODE WALLET


=======================================================
Device: NVIDIA GeForce RTX 4080 SUPER
-------------------------------------------------------
Hashrate: 18415616.000000 H/s
=======================================================
```  

Here is a typical output when NOT mining
```
Bitcoin-PoW GPU Miner v26.5.4

Hash found - NONCE: 839e7000000c4128

Hash no sig low64: 0000000000000000

!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.
!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.
!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.


=======================================================
Device: NVIDIA GeForce RTX 4080 SUPER
-------------------------------------------------------
Hashrate: 0.000000 H/s
=======================================================

```

# Building
[PDCurses is needed](https://github.com/wmcbrine/PDCurses)
```
git clone https://github.com/wmcbrine/PDCurses.git
```

Open an x64 Native Tools command prompt for visual studio and build from wincon folder
```
nmake -f Makefile.vc clean
nmake -f Makefile.vc
```

Look for USERNAME in BTCW_CUDA_MINER.vcxproj using text editor and put correct location for PDCurses paths.  
Build using visual studio