# AI-MONG US: Find the AI-Mpostor

## Overview

AI-MONG US is a project inspired by the popular social deduction game *Among
Us*. It explores the use of artificial intelligence (AI) to tackle games that
incorporate psychological and sociological elements, where players operate
under imperfect and incomplete information. The primary goal is to determine
whether AI can effectively analyze and "solve" such games by simulating
human-like decision-making and interactions.

## Game Description

*Among Us* is a multiplayer social deduction game where players are divided
into **Crewmates** and **Impostors**:

- **Crewmates** aim to complete tasks, identify and eject Impostors, and
  survive.
- **Impostors** aim to eliminate Crewmates, sabotage tasks, and avoid
  detection.

This project replicates the game environment and uses AI agents to simulate
strategies and behaviors for both roles.

## Key Features

### Agents

This project incorporates a range of agents, divided into **AI-Based Agents**
and **Non-AI Agents**. All agents are specialized for the roles of Crewmate and
Impostor, tailoring their strategies to the unique objectives of each role.

#### AI-Based Agents

1. **Neural Network Agents**:
    - Use neural networks to process gameplay data and make decisions based on
      patterns learned from observations.
2. **Probabilistic Graphical Model Agents**:
    - Apply Bayesian reasoning to analyze probabilities and select optimal
      actions in uncertain situations.

#### Non-AI Agents

1. **Logic-Based Decision Tree Agents**:
    - Use predefined rule-based decision trees simulate human decision-making,
      built using prior knowledge of the game. They reflect decisions an
      average player would make in various scenarios, based on group analysis
      and discussions.
2. **Random Agents**:
    - Make moves randomly without considering game state or strategy, serving
      as a baseline for evaluating other agents.

### Search Algorithms

- Implements the A* algorithm for pathfinding and efficient movement across the
  map.

### Visualization

- A `tkinter`-based graphical interface visualizes the game board, agent
  actions, and interactions.

## Code Structure and Key Components

The project is implemented in Python with a modular architecture. Key
components include:

### Core Game Logic

- **`Main.py`**: Entry point for initializing and running the game.
- **`GameFlow.py`**: Base class defining game flow mechanics, such as task
  assignments and turn management.
- **`GameRunner.py`**: Oversees the overall execution of the game,
  orchestrating interactions between components.
- **`GameRunnerHelper.py`**: A utility class supporting `GameRunner.py` with
  various helper functions for processing game settings, agent navigation, and
  observation data.
- **`VoteFlow.py`**: Manages voting logic during discussions, helping agents
  decide whom to vote against.

### Role-Specific Logic

- **`CrewmateGameFlow.py` & `ImpostorGameFlow.py`**: Specialized extensions
  of `GameFlow` for managing Crewmate and Impostor-specific behaviors.

### Agent Framework

- **`Agent.py`**: Base class defining attributes and shared functionalities of
  all agents.
- **`Crewmate.py`** & **`Impostor.py`**: Represent individual entities in the
  game, inheriting common properties and methods from `Agent`.

### Observations and Decision Systems

- **`CrewmateObservations.py` & `ImpostorObservations.py`**: Handle game state
  data and observations for each role.
- **`CrewmateNeuralNetwork.py` & `ImpostorNeuralNetwork.py`**: Implement neural
  network-based decision-making for respective roles.
- **`CrewmatePgmDecisionMaker.py` & `ImpostorPgmDecisionMaker.py`**: Implement
  Bayesian logic for probabilistic decision-making.

### Map and Movement

- **`Board.py`**: Defines the structure of the game map, including walls,
  tasks, and spawn points.
- **`Tile.py`**: Represents individual tiles on the game board with attributes
  such as type and accessibility.
- **`Search.py`**: Implements A* search for optimal pathfinding between
  locations.

### Configuration

- **`Parser.py`**: Parses the `example.json` configuration file containing game
  settings and converts it into a Python dictionary used to create the game
  board.
- **`Constants.py`**: Stores global constants and configuration parameters used
  throughout the project.

### Visualization

- **`BoardGUI.py`**: Provides a visual representation of the game board and
  ongoing interactions.

### Training and Model Storage

- **`crewmate_training/`**: Contains CSV files documenting Crewmate
  observations used for neural network training. These files are dynamically
  generated during execution and should be cleared before retraining.
- **`impostor_training/`**: Holds CSV files documenting Impostor observations
  for training purposes. These files are also generated during execution and
  should be cleared before retraining.
- **`trained nns/`**: Stores binary pickle files of trained neural networks,
  organized by the number of training games conducted. These files are produced
  during execution and can be reused for further refinement.

## Supporting Materials

The **[Project Portfolio](./Project%20Portfolio.docx)** document provides an
in-depth explanation of the project's goals, methodology, challenges, and
results. It serves as a comprehensive resource for understanding the reasoning
and implementation behind the project.

## How to Run

1. **Install Dependencies**:
    - Run the following command to install required Python packages:
      ```bash
      pip install -r requirements.txt
      ```
2. **Configure the Game**:
    - Modify `example.json` to set the number of Crewmates and Impostors and
      specify their AI types (random, logic-based, or neural network).
3. **Run the Game**:
    - Execute the following command to start the simulation:
      ```bash
      python Main.py
      ```
