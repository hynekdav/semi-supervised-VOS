# Label propagation for one-shot video object segmentation

This project was created as a part of master's thesis at the [FEE CTU](https://fel.cvut.cz/en/).

The project is implemented using **Python 3.8.5** and with **PyTorch 1.8.1**. Other used libraries can be found in the **requirements.txt** file. The original code accompanying the [Zhang et al.](https://arxiv.org/abs/2004.07193) can be found on Microsoft's GitHub [repository](https://github.com/microsoft/transductive-vos.pytorch).

## Installation
First of all [Python](https://www.python.org) version at least **3.8** must be installed. Then install [pip](https://pip.pypa.io/en/stable/), which is required to install the project's dependencies. Run following command to install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage
The main application entrypoint **main.py** supports 4 basic commands:
1. train
2. inference
3. validation
4. evaluation

The application is written using [Click](https://click.palletsprojects.com/) library, so every command has automatically generated help pages. Each can be invoked by running:
```bash
python main.py <command> --help}.
```
Examples of running each command are in file **example.sh**.

Additionally, the project offers various visualizations of predicted frames. The visualizations are invoked same way as the main entrypoint and are located in file **visualization.py**:
```bash
python visualization.py <command> --help}.
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Author
Created by [Hynek Davídek](mailto:davidhyn@fel.cvut.cz).


## License
[MIT](LICENSE)