# Ryan Engel
# Chess Engine Version 4


### Necessary Imports ###
import io
import os
import chess
import chess.pgn
import chess.polyglot
import chess.syzygy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import zstandard as zstd
import pygame
import pickle
import time


### Create Game Board ###

# Constants for the board dimensions
WIDTH, HEIGHT = 405, 405
SQUARE_SIZE = WIDTH // 9 
PIECE_SIZE = 45 

# Dictionary for each piece's image
PIECE_IMAGES = {
    'P': 'wp.png',
    'N': 'wn.png',
    'B': 'wb.png',
    'R': 'wr.png',
    'Q': 'wq.png',
    'K': 'wk.png',
    'p': 'bp.png',
    'n': 'bn.png',
    'b': 'bb.png',
    'r': 'br.png',
    'q': 'bq.png',
    'k': 'bk.png'
}

def load_piece_images():
    piece_images = {}
    for piece_name, image_name in PIECE_IMAGES.items():
        image_path = os.path.join('pieces', image_name)
        piece_images[piece_name] = pygame.image.load(image_path).convert_alpha()
    return piece_images

def draw_board(screen, piece_images, board):
    for row in range(9):
        for col in range(9):
            square_color = (255, 206, 158) if (row + col) % 2 == 0 else (209, 139, 71)
            pygame.draw.rect(screen, square_color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            if row < 8 and col < 8:
                square = chess.square(col, 7 - row)
                piece = board.piece_at(square)
                if piece is not None:
                    piece_img = piece_images[piece.symbol()]
                    x_offset = (SQUARE_SIZE - PIECE_SIZE) // 2
                    y_offset = (SQUARE_SIZE - PIECE_SIZE) // 2
                    screen.blit(piece_img, (col * SQUARE_SIZE + x_offset, row * SQUARE_SIZE + y_offset))

            # Fill the rightmost squares with 8 to 1
            if col == 8 and row < 8:
                font = pygame.font.SysFont('Arial', 24)
                text_surface = font.render(str(8 - row), True, (0, 0, 0))  
                screen.blit(text_surface, (col * SQUARE_SIZE + 17, row * SQUARE_SIZE + 7))

            # Fill the bottom row with A to H
            if row == 8 and col < 8:
                font = pygame.font.SysFont('Arial', 24)
                text_surface = font.render(chr(65 + col), True, (0, 0, 0))
                screen.blit(text_surface, (col * SQUARE_SIZE + 16, row * SQUARE_SIZE + 7))

    pygame.display.flip()

def initialize_game(display_number=1, window_width=WIDTH, window_height=HEIGHT):
    from screeninfo import get_monitors
    pygame.init()
    monitors = get_monitors()

    if len(monitors) > 1 and display_number == 1:
        center_x = monitors[1].width // 2 - window_width // 2 + monitors[1].x
        center_y = monitors[1].height // 2 - window_height // 2 + monitors[1].y
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{center_x},{center_y}"  # Positioning to the center of the secondary monitor
    elif display_number == 0:
        center_x = monitors[0].width // 2 - window_width // 2 + monitors[0].x
        center_y = monitors[0].height // 2 - window_height // 2 + monitors[0].y
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{center_x},{center_y}"  # Positioning to the center of the primary monitor

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption('Chess Engine v4')

    return screen




### Mini-Max Algorithm ###

transposition_table = {}
killer_moves = {}

def order_moves(board, moves, depth):
    # Prioritize transposition table move
    tt_entry = transposition_table.get(board.fen())
    if tt_entry:
        tt_move = tt_entry["best_move"]
        if tt_move in moves:
            moves.insert(0, moves.pop(moves.index(tt_move)))

    # Prioritize killer move
    killer = killer_moves.get(depth)
    if killer and killer in moves:
        moves.insert(0, moves.pop(moves.index(killer)))

    return moves

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate(board, board.turn) 

    fen = board.fen()

    if fen in transposition_table and transposition_table[fen]["depth"] >= depth:
        return transposition_table[fen]["value"]

    legal_moves = order_moves(board, list(board.legal_moves), depth)

    if maximizing_player:
        max_eval = float('-inf')
        best_move_for_this_position = None
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move_for_this_position = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                killer_moves[depth] = move
                break
        transposition_table[fen] = {"value": max_eval, "depth": depth, "best_move": best_move_for_this_position}
        return max_eval

    else:
        min_eval = float('inf')
        best_move_for_this_position = None
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move_for_this_position = move
            beta = min(beta, eval)
            if beta <= alpha:
                killer_moves[depth] = move
                break
        transposition_table[fen] = {"value": min_eval, "depth": depth, "best_move": best_move_for_this_position}
        return min_eval

def get_best_move(board, player, time_limit=5):
    best_move = None
    start_time = time.time()

    legal_moves = list(board.legal_moves)

    if not legal_moves:
        print("NO LEGAL MOVES ")
        return None

    # Note: Here is where to implement an opening book, to be implemented later

    depth = 1  
    alpha = float('-inf')
    beta = float('inf')
    maximizing_player = board.turn == player

    while True:
        alpha = float('-inf')
        beta = float('inf')
        best_move_for_depth = None

        for move in legal_moves:
            temp_board = board.copy()
            temp_board.push(move)
            eval = minimax(temp_board, depth, alpha, beta, maximizing_player)

            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit:
                return best_move or best_move_for_depth

            if eval > alpha:
                alpha = eval
                best_move_for_depth = move

        if best_move_for_depth:
            best_move = best_move_for_depth

        depth += 1



### Neural Network Definition ###

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = ChessNN()





### Data Handling ###

def encode_board(board):
    layers = np.zeros((12, 8, 8), dtype=int)
    for square, piece in board.piece_map().items():
        rank = square // 8
        file = square % 8
        index = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        layers[index, rank, file] = 1
    return layers

MAX_GAMES_TO_LOAD = 100

def load_dataset(filename):
    print("Loading Dataset...")
    games_data = []
    games_processed = 0 

    dctx = zstd.ZstdDecompressor()
    with open(filename, 'rb') as compressed_file:
        with dctx.stream_reader(compressed_file) as decompressor, io.TextIOWrapper(decompressor, encoding='utf-8') as pgn_file:
            while True:
                # Control the number of games to load
                if games_processed >= MAX_GAMES_TO_LOAD: 
                    break

                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                games_processed += 1 

                board = game.board()
                result_str = game.headers['Result']
                result = 1 if result_str == "1-0" else 0 if result_str == "0-1" else 0.5
                for move in game.mainline_moves():
                    board.push(move)
                    input_data = torch.tensor(encode_board(board)).float()
                    label = torch.tensor([1.0] if result_str == "1-0" else [0.0] if result_str == "0-1" else [0.5]).float()
                    games_data.append((input_data, label))

    print("Dataset Loaded Successfully!")
    print(f"Total Games Processed: {games_processed}")
    print(f"Total Moves Processed: {len(games_data)}")

    return games_data

### Model Training ###

def train(model, filename, epochs=5):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
    data = load_dataset(filename)

    print("Beginning Training...")
    # Start training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data): 
            inputs = inputs.unsqueeze(0) 
            labels = labels.unsqueeze(0) 
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Check if it's the last batch of the data
            if i == len(data) - 1:
                average_loss = running_loss / (i + 1)  # Calculate the average loss for the epoch
                print(f"Epoch: {epoch + 1} - Loss: {average_loss:.3f}")
                running_loss = 0.0

    print("Finished Training!")

def evaluate(board, player):
    board_representation = encode_board(board)
    with torch.no_grad():
        score = model(torch.tensor(board_representation).float().unsqueeze(0))
    return score.item() if player == chess.WHITE else -score.item()



### AI Game Play ###

def play_game_with_ai(play_itself=True):
    screen = initialize_game(display_number=1, window_width=WIDTH, window_height=HEIGHT)
    piece_images = load_piece_images()
    board = chess.Board()
    player = chess.WHITE

    game_moves = [] 

    running = True
    while running:
        pygame.event.pump()  

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Press 'q' to quit the game
                    running = False

        draw_board(screen, piece_images, board)

        if not board.is_game_over():
            try:
                if player == chess.WHITE:
                    if play_itself:
                        print("White's move (AI):")
                        best_move = get_best_move(board, player)
                        print("BEST MOVE: ", best_move)
                        if best_move:
                            # Store the state, move, and None (because the outcome is unknown yet)
                            game_moves.append((encode_board(board), best_move, None))
                            print(f"AI played (White): {best_move.uci()}")
                            board.push(best_move)
                        else:
                            print("AI failed to determine a best move.")
                            # Check for game-ending conditions
                            if board.is_checkmate():
                                print("Checkmate! Game Over. White wins!" if board.turn == chess.BLACK else "Checkmate! Game Over. Black wins!")
                                running = False
                                break
                            elif board.is_stalemate():
                                print("Stalemate! Game Over.")
                                running = False
                                break
                            elif board.is_insufficient_material():
                                print("Insufficient material! Game Over. White wins!" if board.turn == chess.BLACK else "Insufficient material! Game Over. Black wins!")
                                running = False
                                break
                    else:
                        try:
                            print("White's move:")
                            move_str = input("Enter your move: ")
                            move = chess.Move.from_uci(move_str)
                            if move in board.legal_moves:
                                board.push(move)
                            else:
                                print("Invalid move. Try again.")
                                player = not player
                        except Exception as e:
                            print("Invalid move. Try again.")
                            player = not player
                else:
                    print("Black's move (AI):")
                    best_move = get_best_move(board, player)
                    print("BEST MOVE: ", best_move)
                    if best_move:
                        game_moves.append((encode_board(board), best_move, None))
                        print(f"AI played (Black): {best_move.uci()}")
                        board.push(best_move)
                    else:
                        print("AI failed to determine a best move.")
                        # Check for game-ending conditions
                        if board.is_checkmate():
                            print("Checkmate! Game Over. White wins!" if board.turn == chess.BLACK else "Checkmate! Game Over. Black wins!")
                            running = False
                            break
                        elif board.is_stalemate():
                            print("Stalemate! Game Over.")
                            running = False
                            break
                        elif board.is_insufficient_material():
                            print("Insufficient material! Game Over. White wins!" if board.turn == chess.BLACK else "Insufficient material! Game Over. Black wins!")
                            running = False
                            break

                player = not player  # Switch player turn

                # Check for game-ending conditions
                if board.is_checkmate():
                    print("Checkmate! Game Over. White wins!" if board.turn == chess.BLACK else "Checkmate! Game Over. Black wins!")
                    running = False
                    break
                elif board.is_stalemate():
                    print("Stalemate! Game Over.")
                    running = False
                    break
                elif board.is_insufficient_material():
                    print("Insufficient material! Game Over. White wins!" if board.turn == chess.BLACK else "Insufficient material! Game Over. Black wins!")
                    running = False
                    break

            except ValueError as e:  # To catch specific AI problems
                print(e)
                break
            except KeyboardInterrupt:
                print("\nGame aborted.")
                break
            except Exception as e:  # Catch all other exceptions
                print(f"Error occurred: {e}. Try again.\n")
                break

    pygame.quit()
    return game_moves




### Main Functions ###

def main():
    print("Welcome to Chess AI!")
    print("Choose an option:")
    print("1: Play a game with a saved model")
    print("2: Train a new model")
    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        # Check if the model file exists
        if os.path.exists('chess_engine_model.pth'):
            model.load_state_dict(torch.load('chess_engine_model.pth'))
            model.eval()
            play_itself = False 
            # Set play_itself to false so user can play the AI, if true then AI plays itself; used for testing and evaluation
            play_game_with_ai(play_itself=play_itself)
        else:
            print("No saved model found. Please train a model first.")
    elif choice == '2':
        # Train new model
        dataset_path = "lichess_db_standard_rated_2013-01.pgn.zst"  
        train(model, dataset_path, epochs=5)

        # Save the trained model
        torch.save(model.state_dict(), 'chess_engine_model.pth')
        print("Model trained and saved!")
    else:
        print("Invalid choice.")

if __name__ == '__main__':
    main()

