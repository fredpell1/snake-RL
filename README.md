<!--
**       .@@@@@@@*  ,@@@@@@@@     @@@     .@@@@@@@    @@@,    @@@% (@@@@@@@@
**       .@@    @@@ ,@@          @@#@@    .@@    @@@  @@@@   @@@@% (@@
**       .@@@@@@@/  ,@@@@@@@    @@@ #@@   .@@     @@  @@ @@ @@/@@% (@@@@@@@
**       .@@    @@% ,@@        @@@@@@@@@  .@@    @@@  @@  @@@@ @@% (@@
**       .@@    #@@ ,@@@@@@@@ @@@     @@@ .@@@@@@.    @@  .@@  @@% (@@@@@@@@
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">COMP 490: Honours Project 2023</h3>

  <p align="center">
    Deep Reinforcement Learning for Snake
    <br />
    <a href="https://www.overleaf.com/read/jrytybnkgntp"><strong>Link to (unpublished) paper »</strong></a>
    <br />
    <br />
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#welcome">Welcome</a></li>
    <li><a href="#installing-dependencies">Installing Dependencies</a></li>
    <li><a href="#environment"> Environment </a> </li>
    <li><a href="#models">Models</a></li>
    <li><a href="#training">Training</a></li>
    <li><a href="#testing-and-demo">Testing and Demo</a></li>
  </ol>
</details>

<!-- Welcome -->

## Welcome
My Concordia Honours Project, supervised by [Dr Frederic Godin](https://www.concordia.ca/artsci/math-stats/faculty.html?fpid=frederic-godin).
The purpose of this project was to first apply proven Deep Reinforcement Learning techniques to the game Snake. A secondary goal was to empirically study the impact of the neural net's architecture on performance.

## Installing Dependencies

- python -m pip install -r requirements.txt (if you use pip and venv)
- conda install --file requirements.txt (if you use conda)

## Environment
The environment is implemented with PyGame, gym, and numPy. The source code can be found [here](https://github.com/fredpell1/snake-RL/blob/main/src/envs/snake.py). 

## Models
Several models/approaches were tried, namely TD($\lambda$) learning with Neural Networks, Deep Q-Learning with a feedforward neural network, and Deep Q-Learning with experience replay and with CNNs. You can find the hyperparameters of these models in the [config](https://github.com/fredpell1/snake-RL/tree/main/configs) folder and the implementation [here](https://github.com/fredpell1/snake-RL/tree/main/src/agents). 

## Training
Each model was trained for 50 000 games. The code can be found [here](https://github.com/fredpell1/snake-RL/tree/main/src/utils) and [here](https://github.com/fredpell1/snake-RL/blob/main/src/main.py).

## Testing and Demo
The successful models were then tested for 50 games. You can see the results [here](https://github.com/fredpell1/snake-RL/blob/main/src/plot.ipynb). There's also a CLI to run the model in training, testing, or demo mode.
