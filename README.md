# Abstract
This paper presents a distributed hybrid controller for solving online source seeking problem in a dynamical environment. By formulating the problem as a multiple pursuer multiple evaders game, we simultaneously addresses the problem of discrete assignment of mobile robots to sources as well as the continuous controls strategies for capturing individual sources of external dynamic flow-field.

# Introduction
Source-seeking, sometimes referred as extremum-seeking, involves locating and tracking one or multiple spots, associated with the maximum or minimum value of interest, in a possibly dynamical environment. It has been applied in surveillance, environment monitoring, disaster response etc. For example, consider the scenario where a contamination is introduced in a water body or air through submarines or air-borne drones respectively. Localizing and tracking all sources of contamination as soon as possible can play a crucial role in disseminating early warning as well as in coming up with localized corrective strategies. In this paper, we are assuming that the external flow-field is dynamically changing. Under such circumstances, the mobile robots need to collaboratively assign tasks to each other and localize all the moving sources. We formulate the problem of dynamical source seeking as a multiple pursuer multiple evader (MPME) game where identical pursuers represent mobile robots and evaders represent the sources moving passively in a dynamical flow field \( V_f(z,t)\). 
For this work, we assume all the pursuers know positions of evaders. We further assume that pursuers can talk only to their neighbors. We also assume all players know the external dynamic flow field. The kinodynamic constraints of pursuer \( i \) denoted by \(x_i(t)\) are given as:


$\dot{x}_i = u_i(t) + {V_f}(z,t)$

where \( u_i(t) \) is the control vector. We say that pursuer \(i\) can capture evader \(k\), denoted by \(y_k(t)\) when 

$x_i(t) \in \beta_r(y_k(t)) = {||x-y_k(t)||_2 < r}$

You can find the details on the project website by writing the following command: 
```
python3 -m http.server
```
