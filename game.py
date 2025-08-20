from get_embeddings import load_embeddings

class CodenameGame:
    def __init__(self):
        self.red_team = []
        self.blue_team = []
        self.corpus = load_embeddings()
        self.game_vocab = self.corpus.sample(25)
        self.is_game_over = False

    

    def play(self):
        ...
        return self.is_game_over, ...