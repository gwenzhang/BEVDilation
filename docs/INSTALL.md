**a. Create a conda virtual environment and activate it.**
```shell
conda create -n bevdilation python=3.8 -y
conda activate bevdilation
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch # torhc >= 1.10
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.6.0
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0

```
**e. Install mmdet and mmseg.**
```shell
pip install causal-conv1d==1.1.0
pip install mamba-ssm==1.1.2
```

**f. Clone BEVDilation.**
```
git clone https://github.com/gwenzhang/BEVDilation.git
```

**g. Install BEVdilation**
```shell
cd /path/to/BEVDilation
pip install -v -e .
cd /path/to/BEVDilation/ops_dcnv3
sh ./make.sh
```

**h. Install spconv**
```shell
pip install spconv-cuxxx # select the corresponding cuda, e.g., spconv-cu120
```