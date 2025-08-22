import pandas as pd
import numpy as np
from get_embeddings import load_embeddings
from sentence_transformers import SentenceTransformer


class GameBoard:
    def __init__(self, game_vocab: pd.DataFrame):
        self.words_grid = game_vocab.index.to_numpy().reshape(5, 5)
        self.playable_cells = np.ones((5,5))
        
    def render(self):
        pass

class CodenameGame:
    def __init__(self):
        self.red_team = []
        self.blue_team = []
        self.corpus = load_embeddings()
        self.game_vocab = self.corpus.sample(25)
        self.is_game_over = False

    def initialize_gameboard(self):
        self.game_board = GameBoard(self.game_vocab)

    
    def get_word_embedding(self, word: str):
        assert isinstance(word, str) , 'word must be string'
    
    def play(self):
        ...
        return self.is_game_over, ...

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

if __name__ == '__main__':
    LMmodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    cn = CodenameGame()
    cn.initialize_gameboard()
    print(cn.game_board.words_grid)