import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from utils import Vocab, read_data
"""You should not need any other imports."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, vocab, dims):
        super().__init__()
        self.vocab = vocab
        self.dims = dims
        """TODO: Initialize LSTM weights/layers."""
        V = len(vocab)
        
        # Improved initialization for LSTM
        scale = (2.0 / (V + dims)) ** 0.5
        
        # Input-to-hidden weights for all gates
        self.W_ii = nn.Parameter(torch.empty(dims, V))
        self.W_if = nn.Parameter(torch.empty(dims, V))
        self.W_io = nn.Parameter(torch.empty(dims, V))
        self.W_ig = nn.Parameter(torch.empty(dims, V))
        
        # Hidden-to-hidden weights for all gates
        self.W_hi = nn.Parameter(torch.empty(dims, dims))
        self.W_hf = nn.Parameter(torch.empty(dims, dims))
        self.W_ho = nn.Parameter(torch.empty(dims, dims))
        self.W_hg = nn.Parameter(torch.empty(dims, dims))
        
        # Biases for all gates - initialize forget gate bias to 1 to help with remembering
        self.b_i = nn.Parameter(torch.zeros(dims))
        self.b_f = nn.Parameter(torch.ones(dims))  # Initialize to 1 for better memory
        self.b_o = nn.Parameter(torch.zeros(dims))
        self.b_g = nn.Parameter(torch.zeros(dims))
        
        # Initialize weights properly
        for weight in [self.W_ii, self.W_if, self.W_io, self.W_ig]:
            nn.init.xavier_uniform_(weight)
        for weight in [self.W_hi, self.W_hf, self.W_ho, self.W_hg]:
            nn.init.orthogonal_(weight)  # Orthogonal initialization for recurrent weights
        
        # Hidden-to-output weights
        self.W_ho_out = nn.Parameter(torch.empty(V, dims))
        self.b_o_out = nn.Parameter(torch.zeros(V))
        nn.init.xavier_uniform_(self.W_ho_out)
        
        # Add dropout for regularization (applied to hidden state)
        self.dropout = nn.Dropout(0.2)

    def start(self):
        h = torch.zeros(self.dims, device=device)
        c = torch.zeros(self.dims, device=device)
        return (h, c)

    def step(self, state, idx):
        """TODO: Pass idx through the layers of the model. 
            Return a tuple containing the updated hidden state (h) and cell state (c), and the log probabilities of the predicted next character."""
        h, c = state
        
        # Convert index to one-hot vector
        x = F.one_hot(torch.tensor(idx, device=device), num_classes=len(self.vocab)).float()
        
        # LSTM computations
        i = torch.sigmoid(self.W_ii @ x + self.b_i + self.W_hi @ h)
        f = torch.sigmoid(self.W_if @ x + self.b_f + self.W_hf @ h)
        o = torch.sigmoid(self.W_io @ x + self.b_o + self.W_ho @ h)
        g = torch.tanh(self.W_ig @ x + self.b_g + self.W_hg @ h)
        
        # Update cell state and hidden state
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        # Apply dropout to hidden state for regularization
        h_new = self.dropout(h_new)
        
        # Output computation
        output = self.W_ho_out @ h_new + self.b_o_out
        log_probs = F.log_softmax(output, dim=0)
        
        return (h_new, c_new), log_probs

    def predict(self, state, idx):
        """TODO: Obtain the updated state and log probabilities after calling self.step(). 
            Return the updated state and the most likely next symbol."""
        # Get updated state and log probabilities
        new_state, log_probs = self.step(state, idx)
        
        # Get the index of the most likely next character
        next_idx = torch.argmax(log_probs).item()
        
        # Convert index back to symbol
        next_symbol = self.vocab.denumberize(next_idx)
        
        return new_state, next_symbol

    def fit(self, data, lr=0.002, epochs=25):
        """TODO: This function is identical to fit() from part2.py. 
            The only exception: the state to keep track is now the tuple (h, c) rather than just h. This means after initializing the state with the start state, detatch it from the previous computattion graph like this: `(state[0].detach(), state[1].detach())`"""
        
        # 1. Initialize the optimizer. Use `torch.optim.Adam` with `self.parameters()` and `lr`.
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)  # Added weight decay
        
        # 2. Set a loss function variable to `nn.NLLLoss()` for negative log-likelihood loss.
        loss_fn = nn.NLLLoss()
        
        # Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
        
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
                
                # 1. Initialize the state with the start state, move it to the proper device using `.to(device)`, and detach it from any previous computation graph.
                state = self.start()
                state = (state[0].detach(), state[1].detach())
                
                # 2. Call `optimizer.zero_grad()` to clear any accumulated gradients from the previous update.
                optimizer.zero_grad()
                
                # 3. Initialize a variable to keep track of the loss within a sentence (`loss`).
                loss = 0
                
                # 4. Loop through the characters of the sentence from position 1 to the end (i.e., start with the first real character, not BOS).
                for i in range(1, len(sentence)):
                    
                    # 1. You will need to keep track of the previous character (at position i-1) and current character (at position i). These should be expressed as numbers, not symbols.
                    prev_char = self.vocab.numberize(sentence[i-1])
                    curr_char = self.vocab.numberize(sentence[i])
                    
                    # 2. Call self.step() to get the next state and log probabilities over the vocabulary given the previous character.
                    state, log_probs = self.step(state, prev_char)
                    
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
            
            # 6. For debugging, it will be helpful to print the average loss per character and the runtime after each epoch.
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Time: {elapsed:.2f}s')
            
            # Early stopping check (evaluate on a subset of validation data to save time)
            if epoch % 5 == 0:  # Check every 5 epochs
                val_subset = val_data[:100]  # Use a subset for quick validation
                val_acc = self.evaluate(val_subset)
                print(f'Validation Accuracy (subset): {val_acc:.4f}')
                
                if val_acc >= 0.60:
                    print("✅ Early stopping: Achieved 60% accuracy!")
                    break

    def evaluate(self, data):
        """TODO: Iterating over the sentences in the data, calculate next character prediction accuracy. 
            Use `self.eval()` and `with torch.no_grad()` so that the model is not trained during evaluation.
            Use self.predict() to get the predicted next character, and then check if it matches the real next character found in the data.
            Divide the total correct predictions by the total number of characters to get the final accuracy.
            The code may be identitcal to evaluate() from part2.py."""
        
        # Put model in evaluation mode (disables dropout)
        self.eval()
        
        correct = 0
        total = 0
        
        # Disable gradient computation for evaluation
        with torch.no_grad():
            # Iterate over sentences in data
            for sentence in data:
                # Initialize state
                state = self.start()
                
                # Loop through characters (starting from position 1)
                for i in range(1, len(sentence)):
                    # Get previous and current characters
                    prev_char = sentence[i-1]
                    curr_char = sentence[i]
                    
                    # Get prediction using self.predict()
                    prev_idx = self.vocab.numberize(prev_char)
                    state, predicted_char = self.predict(state, prev_idx)
                    
                    # Check if prediction matches actual character
                    if predicted_char == curr_char:
                        correct += 1
                    
                    total += 1
        
        # Return accuracy
        return correct / total

if __name__ == '__main__':
    
    vocab = Vocab()
    vocab.add('<BOS>')
    vocab.add('<EOS>')
    vocab.add('<UNK>')

    train_data = read_data('data/train.txt')
    val_data = read_data('data/val.txt')
    test_data = read_data('data/test.txt')
    response_data = read_data('data/response.txt')

    for sent in train_data:
        for char in sent:
            vocab.add(char)
    
    print(f"Device: {device}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training samples: {len(train_data)}")
    
    model = LSTMModel(vocab, dims=128).to(device)
    model.fit(train_data, lr=0.002, epochs=25)

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': model.vocab,
        'dims': model.dims
    }, 'lstm_model.pth')

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
    if val_acc >= 0.60 and test_acc >= 0.60:
        print("✅ Successfully achieved 60% accuracy requirement!")
    else:
        print(f"⚠️ Need to achieve 60% accuracy. Current: Val={val_acc:.2%}, Test={test_acc:.2%}")

    """Generate the next 100 characters for the free response questions."""
    print("\n=== Generated Text ===")
    for x in response_data:
        original_prompt = ''.join(x[:-1])  # Convert list to string and remove EOS
        state = model.start()
        
        # Process the entire prompt to update the state
        for char in original_prompt:
            idx = vocab.numberize(char)
            state, _ = model.step(state, idx)
        
        # Start with the original prompt as a string
        current_text = original_prompt
        
        # Generate 100 characters
        for _ in range(100):
            idx = vocab.numberize(current_text[-1])
            state, sym = model.predict(state, idx)
            current_text += sym
        
        # Extract just the generated part (last 100 characters)
        generated_text = current_text[len(original_prompt):]
        
        # Print results
        print('Prompt:', original_prompt[:30] + '...' if len(original_prompt) > 30 else original_prompt)
        print('Generated:', generated_text)
        print()