# mpc_python

Python implementation of mpc controller for path tracking.

## About

The MPC is a model predictive path following controller which does follow a predefined reference path Xref and Yref by solving an optimization problem. The resulting optimization problem is shown in the following equation:

MIN $ J(x(t),U) = \sum^{t+T-1}_{j=t} (x_{j,ref} - x_{j})^T_{j}Q(x_{j,ref} - x_{j}) + u^T_{j}Ru_{j} $

s.t.

$ x(0) = x0 $

$ x_{j+1} = Ax_{j}+Bu_{j})$    for $t< j <t+T-1 $

$ U_{MIN} < u_{j} < U_{MAX} $   for $t< j <t+T-1 $

The vehicle dynamics are described by the differential drive model:

* $\dot{x} = v\cos{\theta}$ 
* $\dot{y} = v\sin{\theta}$
* $\dot{\theta} = w$

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
