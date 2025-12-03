# 6.7920-Final-Project-Racecar


## Install
- edited `/src/f110-gym/setup.py` to make gym and numpy versions more permissive
- this lets me run `pip install -e ./src/f110-gym` and then `pip install -r requirements.txt`
    - BUT this does break `examples/waypoint_follow.py` due to API changes from gym 0.19.0 --> 0.21.0
    - `train.py` seems to run ok though