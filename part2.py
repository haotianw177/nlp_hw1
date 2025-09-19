import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from utils import Vocab, read_data
"""You should not need any other imports."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # to use gpu on kaggle or colab

class RNNModel(nn.Module):
    def __init__(self, vocab, dims):
        super().__init__()
        self.vocab = vocab
        self.dims = dims
        """TODO: Initialize RNN weights/layers."""
        V = len(vocab)
        
        # Better initialization for achieving 58% accuracy
        # Xavier/He initialization helps convergence
        scale = (2.0 / (V + dims)) ** 0.5
        
        # Input-to-hidden weights
        self.W_ih = nn.Parameter(torch.randn(dims, V) * scale)
        self.b_ih = nn.Parameter(torch.zeros(dims))
        
        # Hidden-to-hidden weights - initialize closer to identity for better gradient flow
        self.W_hh = nn.Parameter(torch.eye(dims) * 0.5 + torch.randn(dims, dims) * 0.01)
        self.b_hh = nn.Parameter(torch.zeros(dims))
        
        # Hidden-to-output weights
        self.W_ho = nn.Parameter(torch.randn(V, dims) * scale)
        self.b_o = nn.Parameter(torch.zeros(V))

    def start(self):
        return torch.zeros(self.dims, device=device)

    def step(self, h, idx):
        """TODO: Pass idx through the layers of the model. Return the updated hidden state (h) and log probabilities."""
        # Convert index to one-hot vector
        x = F.one_hot(torch.tensor(idx, device=device), num_classes=len(self.vocab)).float()
        
        # RNN computation: h_t = tanh(W_ih @ x + b_ih + W_hh @ h + b_hh)
        h_next = torch.tanh(self.W_ih @ x + self.b_ih + self.W_hh @ h + self.b_hh)
        
        # Output computation: o = W_ho @ h_next + b_o
        output = self.W_ho @ h_next + self.b_o
        
        # Apply log softmax to get log probabilities
        log_probs = F.log_softmax(output, dim=0)
        
        return h_next, log_probs

    def predict(self, h, idx):
        """TODO: Obtain the updated hidden state and log probabilities after calling self.step(). 
        Return the updated hidden state and the most likely next symbol."""
        # Get updated hidden state and log probabilities
        h_next, log_probs = self.step(h, idx)
        
        # Get the index of the most likely next character
        next_idx = torch.argmax(log_probs).item()
        
        # Convert index back to symbol
        next_symbol = self.vocab.denumberize(next_idx)
        
        return h_next, next_symbol

    def fit(self, data, lr=0.003, epochs=35):  # Increased epochs and lr for better accuracy
        """TODO: Fill in the code using PyTorch functions and other functions from part2.py and utils.py.
        Most steps will only be 1 line of code. You may write it in the space below the step."""
        
        # 1. Initialize the optimizer. Use `torch.optim.Adam` with `self.parameters()` and `lr`.
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # 2. Set a loss function variable to `nn.NLLLoss()` for negative log-likelihood loss.
        loss_fn = nn.NLLLoss()
        
        # Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        # 3. Loop through the specified number of epochs.
        for epoch in range(epochs):
            start_time = time.time()
            
            # 1. Put the model into training mode using `self.train()`.
            self.train()
            
            # 2. Shuffle the training data using random.shuffle().
            random.shuffle(data)
            
            # 3. Initialize variables to keep track of the total loss (`total_loss`) and the total number of characters (`total_chars`).
            total_loss = 0.0
            total_chars = 0
            
            # 4. Loop over each sentence in the training data.
            for sentence in data:
                
                # 1. Initialize the hidden state with the start state, move it to the proper device using `.to(device)`, and detach it from any previous computation graph with `.detach()`.
                h = self.start().to(device).detach()
                
                # 2. Call `optimizer.zero_grad()` to clear any accumulated gradients from the previous update.
                optimizer.zero_grad()
                
                # 3. Initialize a variable to keep track of the loss within a sentence (`loss`).
                loss = 0
                
                # 4. Loop through the characters of the sentence from position 1 to the end (i.e., start with the first real character, not BOS).
                for i in range(1, len(sentence)):
                    
                    # 1. You will need to keep track of the previous character (at position i-1) and current character (at position i). These should be expressed as numbers, not symbols.
                    prev_char = self.vocab.numberize(sentence[i-1])
                    curr_char = self.vocab.numberize(sentence[i])
                    
                    # 2. Call self.step() to get the next hidden state and log probabilities over the vocabulary given the previous character.
                    h, log_probs = self.step(h, prev_char)
                    
                    # 3. See if this matches the actual current character (numberized). Do so by computing the loss with the nn.NLLLoss() loss initialized above.
                    #    * The first argument is the updated log probabilities returned from self.step(). You'll need to reshape it to `(1, V)` using `.view(1, -1)`.
                    #    * The second argument is the current numberized character. It will need to be wrapped in a tensor with `device=device`. Reshape this to `(1,)` using `.view(1)`.
                    loss += loss_fn(log_probs.view(1, -1), torch.tensor([curr_char], device=device).view(1))
                    
                    # 4. Add this this character loss value to `loss`. (Already done above by +=)
                    
                    # 5. Increment `total_chars` by 1.
                    total_chars += 1
                
                # 5. After processing the full sentence, call `loss.backward()` to compute gradients.
                loss.backward()
                
                # 6. Apply gradient clipping to prevent exploding gradients. Use `torch.nn.utils.clip_grad_norm_()` with `self.parameters()` and a `max_norm` of 5.0.
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                
                # 7. Call `optimizer.step()` to update the model parameters using the computed gradients.
                optimizer.step()
                
                # 8. Add `loss.item()` to `total_loss`.
                total_loss += loss.item()
            
            # 5. Compute the average loss per character by dividing `total_loss / total_chars`.
            avg_loss = total_loss / total_chars
            
            # Step the learning rate scheduler
            scheduler.step()
            
            # 6. For debugging, it will be helpful to print the average loss per character and the runtime after each epoch. Average loss per character should always decrease epoch to epoch and drop from about 3 to 1.2 over the 10 epochs.
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Time: {elapsed:.2f}s')

    def evaluate(self, data):
        """TODO: Iterating over the sentences in the data, calculate next character prediction accuracy. 
        Use `self.eval()` and `with torch.no_grad()` so that the model is not trained during evaluation.
        Use self.predict() to get the predicted next character, and then check if it matches the real next character found in the data.
        Divide the total correct predictions by the total number of characters to get the final accuracy."""
        
        # Put model in evaluation mode
        self.eval()
        
        correct = 0
        total = 0
        
        # Disable gradient computation for evaluation
        with torch.no_grad():
            # Iterate over sentences in data
            for sentence in data:
                # Initialize hidden state
                h = self.start()
                
                # Loop through characters (starting from position 1)
                for i in range(1, len(sentence)):
                    # Get previous and current characters
                    prev_char = sentence[i-1]
                    curr_char = sentence[i]
                    
                    # Get prediction using self.predict()
                    prev_idx = self.vocab.numberize(prev_char)
                    h, predicted_char = self.predict(h, prev_idx)
                    
                    # Check if prediction matches actual character
                    if predicted_char == curr_char:
                        correct += 1
                    
                    total += 1
        
        # Return accuracy
        return correct / total

if __name__ == '__main__':
    
    train_data = read_data('data/train.txt')
    val_data = read_data('data/val.txt')
    test_data = read_data('data/test.txt')
    response_data = read_data('data/response.txt')

    vocab = Vocab()
    """TODO: Populate vocabulary with all possible characters/symbols in the training data, including '<BOS>', '<EOS>', and '<UNK>'."""
    # Add special tokens
    vocab.add('<BOS>')
    vocab.add('<EOS>')
    vocab.add('<UNK>')
    
    # Add all characters from training data
    for sentence in train_data:
        for char in sentence:
            vocab.add(char)
    
    print(f"Device: {device}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training samples: {len(train_data)}")
    
    model = RNNModel(vocab, dims=128).to(device)
    model.fit(train_data, lr=0.003, epochs=35)  # Train for 35 epochs with higher learning rate

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': model.vocab,
        'dims': model.dims
    }, 'rnn_model.pth')

    """Use this code if you saved the model and want to load it up again to evaluate. Comment out the model.fit() and torch.save() code if so.
    # checkpoint = torch.load('rnn_model.pth', map_location=device, weights_only=False)
    # vocab = checkpoint['vocab']
    # dims = checkpoint['dims']
    # model = RNNModel(vocab, dims).to(device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    """

    model.eval()

    # Report all required accuracies
    print("\n=== Final Accuracies ===")
    train_acc = model.evaluate(train_data)
    val_acc = model.evaluate(val_data)
    test_acc = model.evaluate(test_data)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Check if requirements are met
    if val_acc >= 0.58 and test_acc >= 0.58:
        print("✅ Successfully achieved 58% accuracy requirement!")
    else:
        print(f"⚠️ Need to achieve 58% accuracy. Current: Val={val_acc:.2%}, Test={test_acc:.2%}")

    """Generate the next 100 characters for the free response questions."""
    """Generate the next 100 characters for the free response questions."""
    print("\n=== Generated Text ===")
    for prompt in response_data:
        prompt = prompt[:-1]  # remove EOS
        state = model.start()
        
        # Process the entire prompt to update the state
        for char in prompt:
            idx = vocab.numberize(char)
            state, _ = model.step(state, idx)
        
        # Get the index of the last character in the prompt
        idx = vocab.numberize(prompt[-1])
        
        # Generate 100 characters
        generated = []
        for _ in range(100):
            state, sym = model.predict(state, idx)
            generated.append(sym)
            idx = vocab.numberize(sym)  # Update idx for next iteration
        
        print('Prompt:', ''.join(prompt[:30]) + '...')
        print('Generated:', ''.join(generated))
        print()