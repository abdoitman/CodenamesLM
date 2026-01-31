# CodenamesLM
**Language Model Agents Competing in the Board Game Codenames**

An implementation of an autonomous Codenames game where language model-powered AI agents compete against each other. The game uses semantic embeddings to understand word relationships and make strategic decisions in this classic word association board game.

## Overview

This project implements a complete Codenames game environment with AI agents that use pre-trained language models (Sentence Transformers) and FAISS for semantic similarity searching. Two teams (Blue and Red) compete, each with a **Spymaster** (gives clues) and **Field Operative** (guesses words).

### Game Rules
- **Board**: 5×5 grid of 25 words with hidden color assignments
- **Roles**: Spymaster (gives one-word clues) and Field Operative (guesses words)
- **Goal**: Each team tries to identify all their team's words while avoiding the assassin
- **Win Condition**: First team to identify all their words wins
- **Lose Condition**: A team loses if they guess the assassin (black card)

## Project Structure

```
CodenamesLM/
├── main.py                    # Entry point for the game
├── game.py                    # Core game logic and board management
├── create_word_embeddings.py  # Utility to generate word embeddings
├── get_corpus.py              # Corpus preparation script
├── corpus.csv                 # Dictionary of playable words
├── word_embeddings.index      # Pre-computed FAISS index for word embeddings
├── id_to_word.npy             # Mapping of embedding IDs to words
├── embeddings.json            # Additional embedding storage
├── requirements.txt           # Python dependencies
├── LICENSE                    # License file
└── README.md                  # This file
```

## Getting Started

### Prerequisites
- Python 3.8+
- All dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CodenamesLM.git
cd CodenamesLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the game:
```bash
python main.py
```

## How It Works

### Core Components

#### `game.py` - Game Engine
- **`GameBoard`**: Manages the 5×5 word grid and color assignments
- **`CodenameGame`**: Main game controller handling turn logic, scoring, and win conditions
- **`Spymaster`**: AI agent that generates clues based on semantic similarity
- **`FieldOperative`**: AI agent that interprets clues and guesses words

The game engine:
- Loads a random sample of 25 words from the corpus
- Randomly assigns colors (8 or 9 team words, 7 neutral, 1 assassin per team)
- Manages turn-based gameplay with alternating teams
- Tracks scores and determines victory conditions

#### `main.py` - Game Orchestrator
- Initializes the Sentence Transformer model (`sentence-transformers/all-MiniLM-L6-v2`)
- Loads pre-computed FAISS word embeddings index
- Creates four AI agents (2 per team)
- Orchestrates game flow with configurable rendering options

#### `create_word_embeddings.py` - Embedding Generator
- Builds a corpus from NLTK word lists, WordNet, and Codenames corpus
- Filters by word frequency (>1e-6 in English)
- Generates embeddings using Sentence Transformers
- Stores embeddings in FAISS index for fast similarity search
- Creates ID-to-word mapping for quick lookups

### Semantic Intelligence
The AI agents use:
- **Sentence Transformers**: Pre-trained model to encode words into semantic vectors
- **FAISS (Facebook AI Similarity Search)**: Efficient nearest-neighbor search for finding related words
- **Cosine Similarity**: Distance metric for semantic relatedness

## Current Features
**Implemented:**
- Two-team game with alternating turns
- Spymaster and Field Operative roles
- Semantic word similarity search using FAISS
- Automatic win/loss detection
- Color-coded console output
- Game board state management
- Score tracking
- Pre-computed word embeddings for fast inference

## Future Work & Roadmap

### Phase 1: Enhanced AI Strategy
- [ ] **Advanced Spymaster Logic**: Improve clue generation
  - Incorporate BabelNet or ConceptNet for richer semantic relations.

- [ ] **Field Operative Confidence Scoring**: Better guess selection
  - Implement confidence thresholds before guessing
  - Allow uncertainty in decision-making
  - Handle edge cases (no strong connection found)

### Phase 2: Game Improvements
- [ ] **Difficulty Levels**: Add configurable AI skill levels
  - Easy: Random guess selection with low strategy
  - Medium: Basic semantic similarity matching
  - Hard: Advanced clustering and risk assessment

- [ ] **Replay System**: Save and analyze game logs
  - Store clue/guess history
  - Calculate win rates by clue type
  - Analyze agent performance metrics

- [ ] **Human Player Integration**: Allow human participation
  - Interactive command-line interface for player input
  - Spectator mode for human observation
  - Manual clue input from human Spymasters

### Phase 3: Model & Evaluation
- [ ] **Alternative Embedding Models**: Experiment with different representations
  - GPT-based embeddings (e.g., OpenAI embeddings)
  - Domain-specific Codenames embeddings
  - Multilingual models for other language support

- [ ] **Systematic Evaluation Framework**
  - Benchmark different agent strategies
  - Track win rates, average clues per round, guess accuracy
  - Compare model performance across game variations

- [ ] **Learning & Adaptation**: Improve agents over time
  - Fine-tune embeddings based on game outcomes
  - Implement reinforcement learning for strategy optimization
  - Learn human-like cluing patterns from datasets

### Phase 4: Scalability & Production
- [ ] **Web Interface**: Browser-based game viewer
  - Real-time game board visualization
  - Live game replay with detailed analytics
  - Multi-game tournament system

- [ ] **Database Integration**: Persistent storage
  - Store game history and statistics
  - Track agent performance over time
  - Enable leaderboards and comparisons

- [ ] **API Server**: REST/WebSocket backend
  - Host the game engine as a service
  - Support multiple concurrent games
  - Enable integration with other platforms

- [ ] **Configuration System**: Customizable game settings
  - Corpus selection (different word sets)
  - Model selection and hyperparameter tuning
  - Agent personality customization

## Performance Metrics

Future metrics to track:
- **Win Rate**: Percentage of games won per team
- **Turns to Victory**: Average number of turns needed to win
- **Clue Efficiency**: Average words guessed correctly per clue
- **Error Rate**: Percentage of incorrect guesses
- **Model Latency**: Clue generation and guess time

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request
   
Please ensure your code adheres to the existing style and includes appropriate tests.
