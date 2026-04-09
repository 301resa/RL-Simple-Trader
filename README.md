Simple Trader Reinforcement learning Agent


main.py
========
Entry point for the RL Trading Agent.

Usage:
    # Train from scratch
    python main.py --mode train --config config/ --data data

    # Continue training from checkpoint
    python main.py --mode train --config config/ --data data/ --checkpoint logs/checkpoints/best_model.zip

    # Evaluate (backtest) a trained model
    python main.py --mode evaluate --config config/ --data data/ \\
                   --checkpoint logs/checkpoints/best_model.zip

    # Print journal analysis for a completed backtest
    python main.py --mode analyse --journal logs/journal/

    # Walk-forward analysis (rolling 12-month train / 5-week val folds)
    python main.py --mode walk_forward --config config/ --data data/

All parameters are loaded from YAML config files — no command-line
parameter overrides for model hyperparameters (edit the YAML instead).
"""

