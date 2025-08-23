import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import random
import faiss
from itertools import cycle
from nltk.stem import PorterStemmer

class GameBoard:
    def __init__(self, game_vocab: pd.DataFrame):
        self.word_list = game_vocab.values.flatten().tolist()
        self.playable_cards = dict(zip(self.word_list, [1] * 25))
        self.words_grid = game_vocab.to_numpy().reshape(5, 5)
        
class CodenameGame:
    def __init__(self):
        self.score = {'red' : 0, 'blue': 0}
        self.__corpus = pd.read_csv('corpus.csv')
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
        self.__words_count = {
            'blue' : game_assignment.count('blue'),
            'red' : game_assignment.count('red')
        }
   
    def get_words_list(self):
        return self.__words_list

    def set_score(self, team: str):
        self.score[team] += 1

    def disable_card(self, word: str):
        self.game_board.playable_cards[word] = 0

    def evaluate_guess(self, guess: str, team: str):
        color = self.key_card[guess]
        
        if color == 'white': 
            print(f"{team.title()} field operative guessed '{guess}' which is a civilian (white card)")
        elif color == 'black':
            self.is_game_over = True
            print(f"{team.title()} field operative guessed '{guess}' which is the assassin (black card)")
            print(f"TEAM {team.capitalize()} LOST!!")
        elif color == team:
            print(f"{team.title()} field operative guessed '{guess}' which is correct ({team} agent)")
            self.set_score(team)
        else:
            print(f"{team.title()} field operative guessed '{guess}' which is not correct ({color} agent)")
            self.set_score(color)
        
        self.disable_card(guess)
        return color
    
    def check_score(self):
        if self.score['red'] == self.__words_count['red']: print('RED TEM WON !!')
        elif self.score['blue'] == self.__words_count['blue']: print('BLUE TEM WON !!')
        else: return
        self.is_game_over = True
        print(f"Blue team scored {self.score['blue']} points")
        print(f"Red team scored {self.score['red']} points")

    def play(self, blue_team: tuple, red_team: tuple, render: bool = False):
        if self.starting_team == 'red' : take_turns = cycle([('red', red_team), ('blue', blue_team)])
        else: take_turns = cycle([('blue', blue_team), ('red', red_team)])
        while not self.is_game_over:
            if render: self.render()
            for team, (spymaster, field_operative) in take_turns:
                print(f"{team.title()} turn's started.")

                clue, num_of_words = spymaster.give_clue()
                print(f"{team.title()} spymaster's clue : {clue} for {num_of_words} cards")
                guesses = field_operative.guess(clue= clue, num_of_words= num_of_words)
                for guess in guesses:
                    color = self.evaluate_guess(guess, team)
                    if color != team: break

                print(f"{team.title()} turn's ended.")
                if not self.is_game_over: self.check_score() # This won't be the case if some player chose the assassin word
        if render: self.close()

    def render(self):
        pass

    def close(self):
        pass

class Player:
    def __init__(self, game: CodenameGame, team: str, model, index: faiss.IndexFlatIP | faiss.IndexFlatL2):
        self.team = team
        self.LM = model
        self.corpus_index = index
        self.game = game
        self.gameboard = game.game_board

class Spymaster(Player):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__key_card = self.game.key_card
        self.__stemmer = PorterStemmer()

    def give_clue(self) -> tuple[str, int]:
        ## Note that we can't give a clue that is derived from a word on the board
        ## Use a stemmer to remove these kinds of words
        playable_words_list = [k for k, v in self.gameboard.playable_cards.items() if v == 1]
        playable_words_roots_list = [self.__stemmer.stem(word) for word in playable_words_list]
        allie_cards = [k for k, v in self.__key_card.items() if (v == self.team) and (k in playable_words_list)]
        white_cards = [k for k, v in self.__key_card.items() if (v == 'white') and (k in playable_words_list)]
        black_cards = [k for k, v in self.__key_card.items() if (v == 'black') and (k in playable_words_list)]
        enemy_cards = [k for k, v in self.__key_card.items() if k not in allie_cards + white_cards + black_cards]

        allie_cards_embeddings = self.LM.encode(allie_cards, convert_to_numpy= True).astype("float32")
        enemy_cards_embeddings = self.LM.encode(enemy_cards, convert_to_numpy= True).astype("float32")
        white_cards_embeddings = self.LM.encode(white_cards, convert_to_numpy= True).astype("float32")
        black_cards_embeddings = self.LM.encode(black_cards, convert_to_numpy= True).astype("float32")

        ### Here comes the lovely logic

        clue = ''
        num_of_words = 0
        return (clue, num_of_words)

    
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