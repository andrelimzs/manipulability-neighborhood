# manipulability-neighborhood

## Objective
Predict the manipulability neighborhood of a given joint configuration.

## Getting Started
Install dependencies in `requirements.txt`

## Usage
### Generate Dataset
To generate 100000 training examples and 20000 validation examples
```
python generate_dataset.py --trainN 100000 --validN 20000 -o dataset.npz
```


### Train Model
To train the model for 200 epochs and save to `model.pt`
```
python learn_manipulability.py --epochs 200 --model model.pt
```

### Plot Prediction
To plot the model saved in default location
```
python Plotting.py
```

To plot other models
```
python Plotting.py --model <path_to_model>
```
