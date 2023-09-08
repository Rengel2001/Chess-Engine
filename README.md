# Chess-Engine
This is a custom designed Chess Engine that uses a combination of the mini-max algorithm, and a convolutional neural network to analyze a chess board and make an intelligent move. When running the program, the player can play against this engine to evaluate its performance.

## Instructions
To test the engine yourself, clone this repository and in your terminal run the python script. When you run the script, you can choose to train a new model, or play against the current model saved in this repository. This way if you would like to make any changes, you can easily retrain the model. To make a move on the chess board use standard coordinate notation.  The required installations can be found in the requirements.txt file.

## Version
This version is currently version number 4. In past versions I have tried other algorithms and architectures of the model, and compared which ones work better. There will likely be further improvements and stronger versions of this engine. 

## Algorithm Design

### Mini-Max
This engine implements the mini-max algorithm. Minimax is a decision-making algorithm, typically used in two-player games, to determine the best move for a player. It examines all possible moves and constructs a move tree, then it evaluates each move based on the most optimal outcome for each player. The function operates recursively, diving deeper into the move tree until it hits the specified depth or the game ends. For each move, it simulates the move, then recurses to the next depth with the updated board state. If the current player is maximizing, it seeks to maximize the evaluation, otherwise, it aims to minimize.

### Alpha-Beta Pruning
A popular approach to implementing the mini-max algorithm is to use alpha-beta pruning, which is what I decided to implement for increased computational efficiency. Alpha-beta pruning is an enhancement of the mini-max algorithm that avoids examining branches (sub-trees) that don't need to be explored. In essence, it prunes away branches in the move tree that will never be chosen by the mini-max procedure. In the specific implementation within the mini-max function, the algorithm maintains two values, alpha and beta. For the maximizing player, if the current evaluation (eval) is greater than the current alpha, alpha is updated. If at any time beta is less than or equal to alpha, the loop breaks (pruning occurs). The same logic, in reverse, is applied for the minimizing player.

### Transpositional Table
Another technique I have implemented to increase computational efficiency was to use a transpositional table. The transposition table is a form of memoization to store previously computed evaluations of board positions. This avoids re-computing evaluations when a board state is revisited. The mini-max function first checks if the current board's FEN is in the transposition table and if the stored depth is equal or larger than the current depth. If true, it retrieves and returns the stored evaluation, bypassing redundant calculations.

### Move Ordering
Move ordering in game tree search algorithms like mini-max is a technique to improve the efficiency of the search by examining promising moves before less promising ones. This can significantly improve the effectiveness of pruning techniques like alpha-beta pruning because better moves lead to earlier cutoffs. I have implemented this strategy in conjunction with the other techniques mentioned as well, ordering the moves from the transpositional table, and killer moves first.

### Iterative Deepening
In addition to the other features of this algorithm, I have also implemented iterative deepening. Iterative deepening is a search strategy that combines the benefits of depth-first search and breadth-first search. It involves repeatedly applying a depth-limited search with increasing depth limits until a certain condition is met, in this case the condition is a time constraint of 10 seconds. This feature helps to decrease the runtime of the algorithm, making it last only 10 seconds for each move.

## Neural Network
The neural network I have chose to implement for this engine is a simple feed forward convolutional neural network. It uses a convolutional layer to detect local patterns in the chessboard and then two fully connected layers to produce a final evaluation score for the given board state. This neural network approach attempts to capture both the local intricacies and the global dynamics of a chess game, providing a deep understanding of the position. 

### Architecture
The architecture begins with a convolutional layer (conv1), tailored with 12 input channels representing the chess pieces, equally bifurcated between black and white. With 32 output channels, a 3x3 kernel, and a padding of 1, this layer adeptly scans the board to discern patterns, such as threats or piece arrangements. The subsequent fully connected layer (fc1) refines the convolutional layer's output, taking the 32x8x8 input and synthesizing it into a comprehensive 64-dimensional vector. The final layer, fc2, processes this vector into a singular value, symbolizing the board's evaluation score, with higher scores showing a more advantageous position for the white player. During the forward pass process, the board state is first relayed through conv1, undergoing a ReLU activation transformation. Post this, the outcome traverses the fc1 layer, which also utilizes ReLU activation. The result from fc1 is then channeled through fc2 to produce the ultimate score. ChessNN's evaluation function transmutes the board for network compatibility, deriving a score thereafter. This score, contingent on the player's color, either remains unchanged for white or is negated for black, preserving the player's perspective in the evaluation.

### Data
The data used to train this neural network was take from the lichess open database which can be found here: https://database.lichess.org/. Furthermore, I used the dataset from january 2013, however I did not use all of this data to train the network due to computational cost. The dataset is decompressed using the Zstandard decompressor, and games are iteratively read from the file. Every game's result is translated into a numeric value, with each move of the game being converted into a tensor representation using the encode_board function. This transformed data is appended to the games_data list as pairs of board states and their corresponding results. 

### Training
The model training begins with the train function, which sets the loss criterion to Mean Squared Error (MSE) and employs the Stochastic Gradient Descent (SGD) optimizer for the training process. The function starts by loading the dataset using the load_dataset function described earlier. The training loop spans across the specified number of epochs. Within each epoch, for every board state and its corresponding label in the dataset, the model processes the input, computes the loss against the true label, and backpropagates the error to update model parameters. The loss is accumulated and reported every 100 iterations. The evaluate function later uses the trained model to evaluate a given chess board state and returns a score based on the perspective of the current player.

## Results
As a result of this approach, I have created a chess engine that makes intelligent moves and has some concept of strategic board positioning, but has not yet surpassed my own chess playing ability. The reason for this was due to the limitations described in the next section, along with some technical issues. For example, the model seems to ignore obviously good moves, and furthermore cannot generalize very well when I make unexpected moves. However, one positive result is that it is able to make intelligent moves when it comes to capturing pieces, and gaining strategic positioning on the board. In other words, it is able to get itself into good positions, but executing good moves from these positions is where it is lacking. Overall, this version is significantly better than the last version, which seemed to making random moves with no thought at all. The addition of the neural network in this version made it more capable of understanding patterns on the board that may lead to a win. I think that with more fine-tuning of hyperparameters, and added heuristics to the mini-max algorithm, I may be able to combat these issues and make it a better engine.

## Limitations
One major limitation of this project is GPU usage. To run all of the computations of this model, including both the algorithm, and the neural network training, I used my computer's CPU. Using only the specifications of my laptop this is clearly not enough computing power to make a state-of-the-art chess engine. Another limitation was the complexity of the project itself. With many possible approaches, it was very challenging finding and implementing an effective strategy for such a complex task.

## Future Research
In future versions of this work, I plan on applying a similar technique used in the AlphaZero engine by Google Deepmind. This is similar to my current strategy, but they use monte-carlo tree search instead of the mini-max algorithm and combined with a deep learning network. AlphaZero achieved superhuman performance in chess, surpassing the capabilities of specialized chess engines like Stockfish. AlphaZero's effective use of reinforcement learning has inspired me to implement this technique in the next version.

