### Contain the entry point for the game
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from game import CodenameGame, Spymaster, FieldOperative

if __name__ == '__main__':
    LMmodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    index = faiss.read_index("word_embeddings.index")
    id_to_word = np.load("id_to_word.npy", allow_pickle=True).item()
    
    game = CodenameGame()
    key_card = game.key_card
    cards = game.get_word_list()

    blue_SM = Spymaster(key_card= key_card,
                        team = 'blue',
                        model= LMmodel,
                        index= index)
    blue_FO = FieldOperative(cards= cards,
                             team = 'blue',
                             model= LMmodel,
                             index= index)
    red_SM = Spymaster(key_card= key_card,
                        team = 'red',
                        model= LMmodel,
                        index= index)
    red_FO = FieldOperative(cards= cards,
                             team = 'red',
                             model= LMmodel,
                             index= index)
    
    game.play(
        blue_team= (blue_SM, blue_FO),
        red_team= (red_SM, red_FO),
        render= True
    )