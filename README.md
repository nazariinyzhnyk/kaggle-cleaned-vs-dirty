# SoftServe DS Hackathon 2020

Binary classification Computer Vision problem: 
classify whether plate is clean or dirty. [Kaggle competition](https://www.kaggle.com/c/platesv2/overview).

## Running pipeline

To run the pipeline run main.py.

 
## Requirements

To install required packages run:

```bash
pip install -r requirements.txt
```

## Data

To download data please join competition and run 

```bash
kaggle competitions download -c platesv2
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## TODOs

- prepare basic pipeline +
- integrate tensorboard +
- implement model saving +
- implement inference 
- implement saving/loading of all of the hyperparams to config file
- remove artifacts from photos after bg removal
- documentation on functions
