import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import random
import faiss
from itertools import cycle
from nltk.stem import PorterStemmer
from time import sleep

class GameBoard:
    def __init__(self, game_vocab: pd.DataFrame, key_card: dict):
        self.word_list = game_vocab.values.flatten().tolist()
        self.playable_cards = dict(zip(self.word_list, [1] * 25))
        self.words_grid = game_vocab.to_numpy().reshape(5, 5)
        self.key_card = key_card

    def print_grid(self):
        color_map = {
            'red': '\033[91m',    # Red
            'blue': '\033[94m',   # Blue
            'white': '\033[97m',  # White
            'black': '\033[90m',  # Grey/Black
        }
        reset = '\033[0m'
        for row in self.words_grid:
            display_row = []
            for word in row:
                if self.playable_cards.get(word, 0) == 1:
                    color = color_map.get(self.key_card.get(word, 'white'), reset)
                    display_row.append(f"{color}{word:>10}{reset}")
                else:
                    display_row.append(f"{'---':>10}")
            print(" | ".join(display_row))
        print()
        
class CodenameGame:
    def __init__(self):
        self.score = {'red' : 0, 'blue': 0}
        self.__corpus = pd.read_csv('corpus.csv')
        self.__game_vocab = self.__corpus.sample(25)
        self.is_game_over = False
        self.is_turn_over = False
        self.__words_list = self.__game_vocab.values.flatten().tolist()
        random.shuffle(self.__words_list)
        assignments = [
            ((["black"] * 1 +
            ["white"] * 7 +
            ["blue"] * 8 +
            ["red"] * 9), 'red') ,
            ((["black"] * 1 +
            ["white"] * 7 +
            ["red"] * 8 +
            ["blue"] * 9), 'blue')
        ]
        game_assignment, self.starting_team = random.choice(assignments)
        self.key_card = dict(zip(self.__words_list, game_assignment))
        self.game_board = GameBoard(self.__game_vocab, self.key_card)
        self.__words_count = {
            'blue' : game_assignment.count('blue'),
            'red' : game_assignment.count('red')
        }
        # encode the 25 words on the board
        LM = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        board_embeddings = LM.encode(self.__words_list, convert_to_numpy= True).astype("float32")
        dim = board_embeddings.shape[1]
        self.board_index = faiss.IndexFlatIP(dim)
        self.board_index.add(board_embeddings)
        self.id_to_word_board = {i: w for i, w in enumerate(self.__words_list)}

    def get_words_list(self):
        return self.__words_list

    def set_score(self, team: str):
        self.score[team] += 1

    def disable_card(self, word: str):
        self.game_board.playable_cards[word] = 0

    def evaluate_guess(self, guess: str, team: str):
        color_map = {
            'red': '\033[91m',
            'blue': '\033[94m',
            'white': '\033[97m',
            'black': '\033[90m',
        }
        reset = '\033[0m'
        color = self.key_card[guess]
        colored_guess = f"{color_map.get(color, reset)}{guess}{reset}"
        colored_team = f"{color_map.get(team, reset)}{team.upper()}{reset}"

        if color == 'white':
            print(f"{colored_team} field operative guessed {colored_guess} which is a civilian")
        elif color == 'black':
            self.is_game_over = True
            print(f"{colored_team} field operative guessed {colored_guess} which is the assassin")
            print(f"TEAM {colored_team} LOST!!")
        elif color == team:
            print(f"{colored_team} field operative guessed {colored_guess} which is correct")
            self.set_score(team)
        else:
            print(f"{colored_team} field operative guessed {colored_guess} which is not correct")
            self.set_score(color)

        self.disable_card(guess)
        return color

    def print_score(self):
        color_map = {
            'red': '\033[91m',
            'blue': '\033[94m'
        }
        reset = '\033[0m'
        print(f"SCORE: {color_map.get('blue', reset)}{self.score['blue']}{reset} | {color_map.get('red', reset)}{self.score['red']}{reset}")

    def check_score(self):
        color_map = {
            'red': '\033[91m',
            'blue': '\033[94m'
        }
        reset = '\033[0m'
        if self.score['red'] == self.__words_count['red']: print(f"{color_map.get('red', reset)}{'RED'}{reset} TEAM WON !!")
        elif self.score['blue'] == self.__words_count['blue']: print(f"{color_map.get('blue', reset)}{'BLUE'}{reset} TEM WON !!")
        else: return
        self.is_game_over = True

    def play(self, blue_team: tuple, red_team: tuple, render: bool = False):
        color_map = {
            'red': '\033[91m',
            'blue': '\033[94m',
            'white': '\033[97m',
            'black': '\033[90m',
        }
        reset = '\033[0m'
        print("Welcome to Codenames!")
        print(f"{color_map.get('blue', reset)}{'BLUE'}{reset} agents are : {', '.join([k for k, v in self.key_card.items() if v == 'blue'])}")
        print(f"{color_map.get('red', reset)}{'RED'}{reset} agents are : {', '.join([k for k, v in self.key_card.items() if v == 'red'])}")
        print(f"{color_map.get('white', reset)}{'CIVILIAN'}{reset} words are : {', '.join([k for k, v in self.key_card.items() if v == 'white'])}")
        print(f"{color_map.get('black', reset)}{'ASSASSIN'}{reset} word is : {', '.join([k for k, v in self.key_card.items() if v == 'black'])}\n")
        if self.starting_team == 'red' : take_turns = cycle([('red', red_team), ('blue', blue_team)])
        else: take_turns = cycle([('blue', blue_team), ('red', red_team)])
        while not self.is_game_over:
            if render: self.render()
            for team, (spymaster, field_operative) in take_turns:
                colored_team = f"{color_map.get(team, reset)}{team.upper()}{reset}"
                print(f"{colored_team} turn's started.")
                print("Current Board:")
                self.game_board.print_grid()
                self.print_score()
                clue, num_of_words = spymaster.give_clue()
                print(f"{colored_team} spymaster's clue : {clue} for {num_of_words} card(s)")
                guesses = field_operative.guess(clue= clue, num_of_words= num_of_words)
                for guess in guesses:
                    color = self.evaluate_guess(guess, team)
                    if color != team: break
                
                self.check_score()
                sleep(5)
                if self.is_game_over: break
                else: print(f"{colored_team} turn's ended.\n")
        if render: self.close()

    def render(self):
        pass

    def close(self):
        pass

class Player:
    def __init__(self, game: CodenameGame, team: str, model, index: faiss.IndexFlatIP | faiss.IndexFlatL2, id_to_word = None):
        self.team = team
        self.LM = model
        self.corpus_index = index
        self.game = game
        self.gameboard = game.game_board
        self.id_to_word = id_to_word

class Spymaster(Player):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__key_card = self.game.key_card
        self.__stemmer = PorterStemmer()

    def give_clue(self) -> tuple[str, int]:
        ## Note that we can't give a clue that is derived from a word on the board
        ## Use a stemmer to remove these kinds of words
        playable_words_list = [k.lower() for k, v in self.gameboard.playable_cards.items() if v == 1]
        playable_words_roots_list = [self.__stemmer.stem(word) for word in playable_words_list]
        allie_cards = [k.lower() for k, v in self.__key_card.items() if (v == self.team) and (k.lower() in playable_words_list)]
        white_cards = [k.lower() for k, v in self.__key_card.items() if (v == 'white') and (k.lower() in playable_words_list)]
        black_cards = [k.lower() for k, v in self.__key_card.items() if (v == 'black') and (k.lower() in playable_words_list)]
        enemy_cards = [k.lower() for k, _ in self.__key_card.items() if k.lower() not in allie_cards + white_cards + black_cards]

        allie_cards_embeddings = self.LM.encode(allie_cards, convert_to_numpy= True).astype("float32")
        enemy_cards_embeddings = self.LM.encode(enemy_cards, convert_to_numpy= True).astype("float32")
        white_cards_embeddings = self.LM.encode(white_cards, convert_to_numpy= True).astype("float32")
        black_cards_embeddings = self.LM.encode(black_cards, convert_to_numpy= True).astype("float32")

        ### Here comes the lovely logic
        if len(allie_cards) > 0:
            mean_allie = np.mean(allie_cards_embeddings, axis=0, keepdims=True)
            if len(enemy_cards) > 0:
                mean_enemy = np.mean(enemy_cards_embeddings, axis=0, keepdims=True)
                clue_vector = mean_allie - mean_enemy
            else:
                clue_vector = mean_allie
            if len(white_cards) > 0:
                mean_white = np.mean(white_cards_embeddings, axis=0, keepdims=True)
                clue_vector = clue_vector - 0.5 * mean_white
            if len(black_cards) > 0:
                mean_black = np.mean(black_cards_embeddings, axis=0, keepdims=True)
                clue_vector = clue_vector - mean_black
            faiss.normalize_L2(clue_vector)
            score, indices = self.corpus_index.search(clue_vector, k = 20)
            scores = score[0]
            indices = indices[0]
            sorted_idx = np.argsort(scores)[::-1]

            for idx in sorted_idx:
                clue_word = self.id_to_word[indices[idx]]
                clue_word_root = self.__stemmer.stem(clue_word)
                if (clue_word_root not in playable_words_roots_list + allie_cards + enemy_cards + white_cards + black_cards) and (clue_word not in playable_words_list + allie_cards + enemy_cards + white_cards + black_cards):
                    # Now we have a valid clue word
                    # Next step is to find how many words this clue is related to
                    clue_embedding = self.LM.encode([clue_word], convert_to_numpy= True).astype("float32")
                    faiss.normalize_L2(clue_embedding)
                    score, words = self.game.board_index.search(clue_embedding, k = 10)
                    scores = score[0]
                    board_indices = words[0]
                    sorted_board_idx = np.argsort(scores)[::-1]
                    num_of_words = 0
                    threshold = 0.25
                    for j in sorted_board_idx:
                        word = self.game.id_to_word_board[board_indices[j]]
                        if word.lower() in allie_cards and scores[j] > threshold:
                            num_of_words += 1
                        elif word in enemy_cards + white_cards + black_cards:
                            break
                    if num_of_words > 0:
                        return (clue_word, num_of_words)
        # If no clue is found, return a random word from the corpus with num_of_words = 1
        return ('itman', 1)

    
class FieldOperative(Player):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cards = self.game.get_words_list()
        cards_embeddings = self.LM.encode(self.cards, convert_to_numpy=True).astype("float32")
        dim = cards_embeddings.shape[1]
        self.cards_index = faiss.IndexFlatIP(dim)
        self.cards_index.add(cards_embeddings)
        self.id_to_word = {i: w for i, w in enumerate(self.cards)}
    
    def guess(self, clue: str, num_of_words: int) -> list:
        clue = self.LM.encode([clue], convert_to_numpy= True).astype("float32")
        faiss.normalize_L2(clue)
        score, words = self.cards_index.search(clue, k = 10)
        playable_words_list = [k for k, v in self.gameboard.playable_cards.items() if v == 1]
        scores = score[0]
        indices = words[0]
        sorted_idx = np.argsort(scores)[::-1]

        guesses = []
        for idx in sorted_idx:
            word = self.id_to_word[indices[idx]]
            if word in playable_words_list:
                guesses.append(word)
            if len(guesses) == num_of_words:
                break
        
        return guesses

if __name__ == '__main__':
    # LMmodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # index = faiss.read_index("word_embeddings.index")
    # id_to_word = np.load("id_to_word.npy", allow_pickle=True).item()
    cn = CodenameGame()
    print(cn.game_board.words_grid)