import pandas as pd
import numpy as np
from get_embeddings import load_embeddings
from sentence_transformers import SentenceTransformer
import random
import faiss

class GameBoard:
    def __init__(self, game_vocab: pd.DataFrame):
        self.word_list = game_vocab.index.to_list()
        self.words_grid = game_vocab.index.to_numpy().reshape(5, 5)
        self.playable_cells = np.ones((5,5))

class Player:
    def __init__(self, team: str, model):
        self.team = team
        self.LM = model

class Spymaster(Player):
    def __init__(self, key_card: dict, kwargs):
        super.__init__(self, **kwargs)
        self.__key_card = key_card

    def give_clue(self) -> tuple[str, int]:
        pass

    def update_key_card(self):
        pass
    
class FieldOperative(Player):
    def __init__(self, cards: list, **kwargs):
        super.__init__(self, **kwargs)
        self.cards = cards
    
    def guess(self, clue: str) -> str | list:
        pass

class CodenameGame:
    def __init__(self):
        self.score = {'red' : 0, 'blue': 0}
        self.__corpus = load_embeddings()
        self.__game_vocab = self.__corpus.sample(25)
        self.game_board = GameBoard(self.__game_vocab)
        self.is_game_over = False
        self.is_turn_over = False
        self.__words_list = self.game_board.word_list
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
   
    def get_word_list(self):
        return self.__words_list

    def get_word_embedding(self, word: str):
        assert isinstance(word, str) , 'word must be string'
        return self.__game_vocab.loc[word, :]

    def set_score(self, team: str):
        self.score[team] += 1

    def evaluate_word(self, word: str, team: str):
        color = self.key_card[word]
        if color == 'white':
            self.is_turn_over = True
        elif color == 'black':
            self.is_game_over = True
            print(f"Team {team} Lost!")
        elif color == team:
            self.set_score(team)
        else:
            self.score(color)
            self.is_turn_over = True

    def play(self, blue_team: tuple[Spymaster, FieldOperative], red_team: tuple[Spymaster, FieldOperative]):
        print(f'Team {self.starting_team} is starting...')
        return self.is_game_over, ...

    def render(self):
        pass

if __name__ == '__main__':
    # LMmodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    index = faiss.read_index("word_embeddings.index")
    id_to_word = np.load("id_to_word.npy", allow_pickle=True).item()
    cn = CodenameGame()
    print(cn.key_card)