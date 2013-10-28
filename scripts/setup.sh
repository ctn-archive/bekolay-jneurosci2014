#!/bin/zsh
if [ ! -d "$HOME/nengo-latest" ]; then
    echo "Moving nengo-* to nengo-latest"
    mv $HOME/nengo-* $HOME/nengo-latest
fi
if [ ! -L "$HOME/nengo-latest/trevor" ]; then
    echo "Linking scripts to /nengo-latest/trevor"
    SCRIPTS=$(readlink -f ..)
    ln -s $SCRIPTS $HOME/nengo-latest/trevor
fi
