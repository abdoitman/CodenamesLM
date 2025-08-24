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

    blue_SM = Spymaster(game= game,
                        team= 'blue',
                        model= LMmodel,
                        index= index,
                        id_to_word= id_to_word)
    blue_FO = FieldOperative(game= game,
                             team= 'blue',
                             model= LMmodel,
                             index= index,
                             id_to_word= id_to_word)
    red_SM = Spymaster(game= game,
                       team= 'red',
                       model= LMmodel,
                       index= index,
                       id_to_word= id_to_word)
    red_FO = FieldOperative(game= game,
                            team= 'red',
                            model= LMmodel,
                            index= index,
                            id_to_word= id_to_word)
    
    game.play(
        blue_team= (blue_SM, blue_FO),
        red_team= (red_SM, red_FO),
        render= False
    )