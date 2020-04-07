# mpc_python

Python implementation of mpc controller for path tracking.

## About

The MPC is a model predictive path following controller which does follow a predefined reference path Xref and Yref by solving an optimization problem. The resulting optimization problem is shown in the following equation:

![](img/quicklatex1.gif)

The vehicle dynamics are described by the differential drive model:

![](img/quicklatex2.gif)

The state variables of the model are:

* $x$ coordinate of the robot
* $y$ coordinate of the robot
* $\theta$ heading of the robot

The inputs of the model are:

* $v$ linear velocity of the robot
* $w$ angular velocity of the robot

## Demo

![](img/demo.gif)

To run the demo:

```bash
python3 mpc_demo/main.py
```

## Requirements

```bash
pip3 install --user --requirement requirements.txt
```
