# Constrained EVLearn

# CityLearn Environmenet
We use the CityLearn which is an open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities. A major challenge for RL is to satisfy hard/soft constraints and provide safe actions. In our context, these could be system related constraints, for example charging/discharging the EV outside the charging/discharging bounds.


## Environment Overview

CityLearn includes energy models of buildings and distributed energy resources (DER) including air-to-water heat pumps, electric heaters and batteries. A collection of building energy models makes up a virtual district (a.k.a neighborhood or community). In each building, space cooling, space heating and domestic hot water end-use loads may be independently satisfied through air-to-water heat pumps. Alternatively, space heating and domestic hot water loads can be satisfied through electric heaters. Additionally, we use the EVs which are integrated in the buildings. 

![Citylearn](https://github.com/intelligent-environments-lab/CityLearn/blob/master/assets/images/citylearn_systems.png)
