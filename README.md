# PAC CAN DO !

Maxime and Ian's team for the Pacman Contest. Currently it implements a planning agent with alpha pruning (not a minimax agent, since the API doesn't let us access the other team agents' states).

Originally, we had planned to use Q-learning (kept in `q_learning.py`) to train a model, but due to the large number of possible states and the fact that we can't know exact opponent states our implementation ending up losing regularly to the baseline agent.

Our current implementation consistently wins against the baseline agent.
