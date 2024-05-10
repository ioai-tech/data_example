# IO-DATA Example

## Fetch data

Download data from [here](https://drive.google.com/drive/folders/1xQjkYJfL77e9iPhirie8gfJFGo7pjtbT?usp=sharing)

`*-full.tar.gz` contains full data.
`*-no-depth.tar.gz` contains the data for the no-depth version of the dataset, which take up less storage.

Extarct data:

```
tar -xvf DATA_NAME.tar.gz
```

## Environment

- Ubuntu >=20.04

## Denpendancy

- python3
- pip3
- opencv-python
- pybullet (for mocap visualization)

## Install packages

```bash
sudo apt-get install python3-pip
pip3 install opencv-python
```

## Usage

run the following command
```bash
python3 load_and_visualize.py YOUR_DATA_PATH
# e.g. python3 load_and_visualize.py ~/IO-DATA-EXAMPLE/20240505_kitchen/
```

If you want to visualize the mocap data,

1. you need install pybullet
   ```bash
   pip3 install pybullet
   ```
2. run the following command
   ```bash
   python3 load_and_visualize.py YOUR_DATA_PATH --visualize_mocap
   ```

`

## Output
