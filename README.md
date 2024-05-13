# IO-DATA Example

## Access data

Download sample data from [here](https://drive.google.com/drive/folders/1xQjkYJfL77e9iPhirie8gfJFGo7pjtbT?usp=sharing)

`*_full.tar` contains full data.
`*_no_depth_low_res.tar` contains the data for the no-depth version of the dataset, including low-resolution RGB images, which occupy less storage space.

Extract data:

```
tar -xvf DATA_NAME.tar
```

If you want **more data with more detailed annotations**, please fill the form [here](https://forms.gle/fDdyipTKDZaL34zC6) to join the waiting list.

![waiting_list](asserts/waiting_list_form.png)

## Environment

- Ubuntu >=20.04

## Denpendancy

- python3
- pip3
- opencv-python
- pybullet (for mocap visualization)

## Install packages

```bash
sudo apt install python3-pip
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

## Output

![output](asserts/output.png)

## Copyright Notice

All software is licensed under the MIT License.

All other materials are offered under a CC BY-NC-SA 4.0 license. You are free to use, copy, and redistribute the material for non-commercial purposes provided you give appropriate credit, provide a link to the license, and indicate if changes were made. If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original. You may not use the material for commercial purposes.
