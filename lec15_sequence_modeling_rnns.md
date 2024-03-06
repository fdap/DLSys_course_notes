# Sequence Modeling + RNNs
***

## Sequence Modeling
The input/output pairs are given in a specific sequence, and we need to use the information about this sequence to help us make predictions.

## Recurrent Neural Networks
- introduce the hidden state.
- stacking RNNs to form deep RNNs.
- training RNNs suffers exploding activation/gradients or vanishing activation/gradients.

## Long Short Term Memory RNNs(LSTM)
- LSTM :
    1. Step 1: divide the hidden unit into two components, called the *hidden state* and the *cell state*.
    2. Step 2:  Use a very specific formula to update the hidden state and cell state (throwing in some other names, like “forget gate”, “input gate”, “output gate” for good measure).
- Why do LSTMs work?
    - The key lies in the calculate the *cell state*, “saturating” sigmoid activation at 1 would just pass through the previous *cell state* untouched
